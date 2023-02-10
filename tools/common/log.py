import logging


def setup_default_logging(default_level: int = logging.INFO) -> None:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt=r"%Y-%m-%d %H:%M:%S")
    )
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
