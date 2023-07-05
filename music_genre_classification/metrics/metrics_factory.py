from torchmetrics import F1Score, Metric, Precision, Recall


class MetricsFactory:
    """
    Factory class for creating metrics.
    """

    def build(metrics_config: dict) -> dict[Metric]:
        metrics = {}
        for metric_config in metrics_config:
            metrics[metric_config["name"]] = MetricsFactory._build_metric(metric_config)
        return metrics

    def _build_metric(config: dict) -> Metric:
        if config["name"] == "F1 Score":
            return F1Score(**config.get("args", {}))
        elif config["name"] == "Precision":
            return Precision(**config.get("args", {}))
        elif config["name"] == "Recall":
            return Recall(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown looper type: {config['name']}")
