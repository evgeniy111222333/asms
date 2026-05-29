"""Kafka message key routing for partition ordering."""


class KeyRouter:
    """Routes messages to partitions based on message keys.

    Ensures that messages for the same symbol/strategy are
    processed in order by mapping them to the same partition.
    """

    def __init__(self, num_partitions: int = 6):
        self.num_partitions = num_partitions

    def get_partition(self, key: str) -> int:
        """Get partition number for a message key.

        Args:
            key: Message key (symbol, strategy_id, etc.).

        Returns:
            Partition number.
        """
        return hash(key) % self.num_partitions

    def get_key_for_symbol(self, symbol: str) -> str:
        """Get routing key for a symbol."""
        return f"symbol:{symbol}"

    def get_key_for_strategy(self, strategy_id: str) -> str:
        """Get routing key for a strategy."""
        return f"strategy:{strategy_id}"

    def get_key_for_order(self, order_id: str) -> str:
        """Get routing key for an order."""
        return f"order:{order_id}"


__all__ = ["KeyRouter"]
