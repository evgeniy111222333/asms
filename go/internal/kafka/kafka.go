// Package kafka provides Kafka/Redpanda producer and consumer helpers for ACMS.
package kafka

import (
        "context"
        "encoding/json"
        "fmt"
        "sync"
        "time"

        "github.com/confluentinc/confluent-kafka-go/v2/kafka"
        "go.uber.org/zap"
)

// Producer wraps a Kafka producer with convenience methods.
type Producer struct {
        producer *kafka.Producer
        logger   *zap.Logger
        mu       sync.Mutex
        msgCount int64
        errCount int64
}

// NewProducer creates a new Kafka producer.
func NewProducer(brokers string, logger *zap.Logger) (*Producer, error) {
        p, err := kafka.NewProducer(&kafka.ConfigMap{
                "bootstrap.servers":        brokers,
                "acks":                     "all",
                "retries":                  5,
                "retry.backoff.ms":         500,
                "queue.buffering.max.kbytes": 10240,
                "queue.buffering.max.messages": 100000,
                "linger.ms":                5,
                "batch.num.messages":       10000,
                "compression.type":         "lz4",
                "client.id":                "acms-go",
        })
        if err != nil {
                return nil, fmt.Errorf("failed to create kafka producer: %w", err)
        }

        prod := &Producer{
                producer: p,
                logger:   logger,
        }

        // Start delivery report handler
        go prod.handleDeliveryReports()

        logger.Info("kafka producer created", zap.String("brokers", brokers))
        return prod, nil
}

func (p *Producer) handleDeliveryReports() {
        for e := range p.producer.Events() {
                switch ev := e.(type) {
                case *kafka.Message:
                        if ev.TopicPartition.Error != nil {
                                p.mu.Lock()
                                p.errCount++
                                p.mu.Unlock()
                                p.logger.Error("kafka message delivery failed",
                                        zap.String("topic", *ev.TopicPartition.Topic),
                                        zap.Error(ev.TopicPartition.Error),
                                )
                        } else {
                                p.mu.Lock()
                                p.msgCount++
                                p.mu.Unlock()
                        }
                case kafka.Error:
                        p.logger.Error("kafka producer error", zap.Error(ev))
                }
        }
}

// Publish sends a message to the specified topic.
func (p *Producer) Publish(topic string, key []byte, value interface{}) error {
        data, err := json.Marshal(value)
        if err != nil {
                return fmt.Errorf("failed to marshal message: %w", err)
        }

        msg := &kafka.Message{
                TopicPartition: kafka.TopicPartition{
                        Topic:     &topic,
                        Partition: kafka.PartitionAny,
                },
                Key:   key,
                Value: data,
        }

        if err := p.producer.Produce(msg, nil); err != nil {
                return fmt.Errorf("failed to produce message: %w", err)
        }

        return nil
}

// PublishAsync sends a message asynchronously and calls the callback on delivery.
func (p *Producer) PublishAsync(topic string, key []byte, value interface{}, callback func(error)) error {
        data, err := json.Marshal(value)
        if err != nil {
                return fmt.Errorf("failed to marshal message: %w", err)
        }

        msg := &kafka.Message{
                TopicPartition: kafka.TopicPartition{
                        Topic:     &topic,
                        Partition: kafka.PartitionAny,
                },
                Key:   key,
                Value: data,
        }

        deliveryChan := make(chan kafka.Event, 1)
        if err := p.producer.Produce(msg, deliveryChan); err != nil {
                return fmt.Errorf("failed to produce message: %w", err)
        }

        go func() {
                e := <-deliveryChan
                switch ev := e.(type) {
                case *kafka.Message:
                        if ev.TopicPartition.Error != nil {
                                p.mu.Lock()
                                p.errCount++
                                p.mu.Unlock()
                                if callback != nil {
                                        callback(ev.TopicPartition.Error)
                                }
                        } else {
                                p.mu.Lock()
                                p.msgCount++
                                p.mu.Unlock()
                                if callback != nil {
                                        callback(nil)
                                }
                        }
                }
        }()

        return nil
}

// Stats returns producer statistics.
func (p *Producer) Stats() (msgCount, errCount int64) {
        p.mu.Lock()
        defer p.mu.Unlock()
        return p.msgCount, p.errCount
}

// Flush waits for all messages to be delivered.
func (p *Producer) Flush(timeoutMs int) int {
        return p.producer.Flush(timeoutMs)
}

// Close flushes and closes the producer.
func (p *Producer) Close() {
        p.logger.Info("closing kafka producer, flushing pending messages")
        remaining := p.producer.Flush(10000)
        if remaining > 0 {
                p.logger.Warn("kafka producer flush incomplete", zap.Int("remaining", remaining))
        }
        p.producer.Close()
        p.logger.Info("kafka producer closed")
}

// Consumer wraps a Kafka consumer with convenience methods.
type Consumer struct {
        consumer *kafka.Consumer
        logger   *zap.Logger
        handler  func(topic string, key, value []byte) error
        stopCh   chan struct{}
        running  bool
        mu       sync.Mutex
        msgCount int64
        errCount int64
}

// NewConsumer creates a new Kafka consumer.
func NewConsumer(brokers, groupID string, topics []string, logger *zap.Logger) (*Consumer, error) {
        c, err := kafka.NewConsumer(&kafka.ConfigMap{
                "bootstrap.servers":  brokers,
                "group.id":           groupID,
                "auto.offset.reset":  "earliest",
                "enable.auto.commit": true,
                "session.timeout.ms": 30000,
                "client.id":          "acms-go-consumer",
        })
        if err != nil {
                return nil, fmt.Errorf("failed to create kafka consumer: %w", err)
        }

        if err := c.SubscribeTopics(topics, nil); err != nil {
                c.Close()
                return nil, fmt.Errorf("failed to subscribe to topics: %w", err)
        }

        logger.Info("kafka consumer created",
                zap.String("brokers", brokers),
                zap.String("group", groupID),
                zap.Strings("topics", topics),
        )

        return &Consumer{
                consumer: c,
                logger:   logger,
                stopCh:   make(chan struct{}),
        }, nil
}

// SetHandler sets the message handler function.
func (c *Consumer) SetHandler(handler func(topic string, key, value []byte) error) {
        c.handler = handler
}

// Start begins consuming messages.
func (c *Consumer) Start() error {
        if c.handler == nil {
                return fmt.Errorf("no message handler set")
        }

        c.mu.Lock()
        c.running = true
        c.mu.Unlock()

        go c.consumeLoop()
        return nil
}

func (c *Consumer) consumeLoop() {
        for {
                select {
                case <-c.stopCh:
                        return
                default:
                }

                ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
                msg, err := c.consumer.ReadMessage(5000)
                cancel()

                if err != nil {
                        if kafkaErr, ok := err.(kafka.Error); ok {
                                if kafkaErr.Code() == kafka.ErrTimedOut {
                                        continue
                                }
                        }
                        c.mu.Lock()
                        c.errCount++
                        c.mu.Unlock()
                        c.logger.Error("kafka consumer error", zap.Error(err))
                        continue
                }

                if err := c.handler(*msg.TopicPartition.Topic, msg.Key, msg.Value); err != nil {
                        c.mu.Lock()
                        c.errCount++
                        c.mu.Unlock()
                        c.logger.Error("message handler error", zap.Error(err))
                } else {
                        c.mu.Lock()
                        c.msgCount++
                        c.mu.Unlock()
                }

                _ = ctx
        }
}

// Stats returns consumer statistics.
func (c *Consumer) Stats() (msgCount, errCount int64) {
        c.mu.Lock()
        defer c.mu.Unlock()
        return c.msgCount, c.errCount
}

// Stop gracefully stops the consumer.
func (c *Consumer) Stop() {
        c.mu.Lock()
        c.running = false
        c.mu.Unlock()

        close(c.stopCh)
        c.consumer.Close()
        c.logger.Info("kafka consumer stopped")
}

// CreateTopic creates a Kafka topic if it does not exist.
func CreateTopic(brokers, topic string, partitions, replication int) error {
        admin, err := kafka.NewAdminClient(&kafka.ConfigMap{
                "bootstrap.servers": brokers,
        })
        if err != nil {
                return fmt.Errorf("failed to create admin client: %w", err)
        }
        defer admin.Close()

        ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
        defer cancel()

        maxRetries := 5
        for i := 0; i < maxRetries; i++ {
                _, err = admin.CreateTopics(ctx, []kafka.TopicSpecification{{
                        Topic:             topic,
                        NumPartitions:     partitions,
                        ReplicationFactor: replication,
                }})
                if err == nil {
                        return nil
                }
                time.Sleep(2 * time.Second)
        }

        return fmt.Errorf("failed to create topic %s after %d retries: %w", topic, maxRetries, err)
}

// EnsureTopics creates multiple topics with default settings.
func EnsureTopics(brokers string, topics []string, logger *zap.Logger) {
        for _, topic := range topics {
                err := CreateTopic(brokers, topic, 3, 1)
                if err != nil {
                        logger.Warn("failed to create topic (may already exist)",
                                zap.String("topic", topic),
                                zap.Error(err),
                        )
                } else {
                        logger.Info("created topic", zap.String("topic", topic))
                }
        }
}
