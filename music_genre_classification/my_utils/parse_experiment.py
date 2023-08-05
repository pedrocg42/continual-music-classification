import experiments


def parse_experiment(function):
    def parser(*args, **kwargs):
        experiment = getattr(experiments, kwargs["experiment"])

        return function(*args, **experiment)

    return parser
