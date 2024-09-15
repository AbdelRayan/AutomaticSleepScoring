import logging
import logging.config
import yaml

def logger_setup(log_file="record.log", console_level="INFO"):
    default_config = """
    version: 1
    loggers:
      runtime:
        level: DEBUG
        handlers: [console, file]
        propagate: false

    handlers:
      console:
        class : logging.StreamHandler
        formatter: brief
        level   : {console_level}
        stream  : ext://sys.stdout
      file:
        class : logging.FileHandler
        formatter: default
        mode: a+
        filename: {log_file}

    formatters:
      brief:
        format: '%(message)s'
      default:
        format: '[%(asctime)s] [%(module)s:%(funcName)20s]: %(message)s'
        datefmt: '%H:%M:%S'

    disable_existing_loggers: true

    """

    new_config = default_config.format(log_file=log_file, console_level=console_level)
    # Load config to dict
    new_config = yaml.load(new_config, Loader=yaml.FullLoader)
    # Configure logger with dict
    logging.config.dictConfig(new_config)

    logger = logging.getLogger('runtime')
    
    logger.debug("Runtime log started.")
    logger.debug("Logging to file {0}".format(log_file))
    
    return logger
