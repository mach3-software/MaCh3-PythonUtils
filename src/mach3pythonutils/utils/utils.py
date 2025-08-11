import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install


LOG_LEVELS = {
    "DEBUG" : logging.DEBUG,
    "INFO" : logging.INFO,
    "WARNING" : logging.WARNING,
    "ERROR" : logging.ERROR,
    "CRITICAL" : logging.CRITICAL
}

# Custom theme for MaCh3 logging
MACH3_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "debug": "dim cyan",
    "timestamp": "dim green",
    "level": "bold",
    "path": "dim blue",
    "line_number": "dim magenta",
    "function": "bold blue"
})

def setup_logging(log_level: str = "INFO", fmt: Optional[str] = None, datefmt: Optional[str] = None, 
                 use_rich: bool = True, rich_theme: Optional[Theme] = None):
    """Sets up the logging configuration for the application with optional rich formatting.

    :param log_level: The logging level to set, defaults to "INFO"
    :type log_level: str
    :param fmt: The format for the log messages, defaults to None (only used when rich is disabled)
    :type fmt: Optional[str], optional
    :param datefmt: The format for the date in log messages, defaults to None (only used when rich is disabled)
    :type datefmt: Optional[str], optional
    :param use_rich: Whether to use rich formatting for logs, defaults to True
    :type use_rich: bool, optional
    :param rich_theme: Custom theme for rich logging, defaults to MACH3_THEME
    :type rich_theme: Optional[Theme], optional
    """
    
    # Getting the log level from the dictionary
    log_level_ = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Install rich traceback handler for better error formatting
    install(show_locals=True)
    
    # Use provided theme or default MaCh3 theme
    theme = rich_theme if rich_theme is not None else MACH3_THEME
    
    # Create console with custom theme
    console = Console(theme=theme, force_terminal=True)
    
    # Create rich handler with custom formatting
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
        log_time_format="[%Y-%m-%d %H:%M:%S]"
    )
    
    # Set up the root logger
    logging.basicConfig(
        level=log_level_,
        format="%(message)s",
        handlers=[rich_handler]
    )
    
    # Log a startup message with styling
    logger = logging.getLogger(__name__)
    logger.debug("[bold green]🚀 MaCh3 Python Utils logging initialized![/bold green]")
            
    logging.getLogger().setLevel(log_level_)


def get_styled_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance that's configured for rich formatting.
    
    :param name: Name for the logger, defaults to None (uses calling module)
    :type name: str, optional
    :return: Configured logger instance
    :rtype: logging.Logger
    """
    return logging.getLogger(name)


def log_section_header(logger: logging.Logger, title: str, level: str = "INFO"):
    """Log a styled section header.
    
    :param logger: Logger instance to use
    :type logger: logging.Logger
    :param title: Title of the section
    :type title: str
    :param level: Log level to use, defaults to "INFO"
    :type level: str
    """
    border = "─" * (len(title) + 4)
    log_func = getattr(logger, level.lower())
    log_func(f"[bold cyan]┌{border}┐[/bold cyan]")
    log_func(f"[bold cyan]│ {title} │[/bold cyan]")
    log_func(f"[bold cyan]└{border}┘[/bold cyan]")



def log_success(logger: logging.Logger, message: str):
    """Log a success message with styling.
    
    :param logger: Logger instance to use
    :type logger: logging.Logger
    :param message: Success message
    :type message: str
    """
    logger.info(f"[bold green]✅ {message}[/bold green]")


def log_error(logger: logging.Logger, message: str):
    """Log an error message with styling.
    
    :param logger: Logger instance to use
    :type logger: logging.Logger
    :param message: Error message
    :type message: str
    """
    logger.error(f"[bold red]❌ {message}[/bold red]")


def log_warning(logger: logging.Logger, message: str):
    """Log a warning message with styling.
    
    :param logger: Logger instance to use
    :type logger: logging.Logger
    :param message: Warning message
    :type message: str
    """
    logger.warning(f"[bold yellow]⚠️  {message}[/bold yellow]")

def log_debug(logger: logging.Logger, message: str):
    """Log a debug message with styling.
    
    :param logger: Logger instance to use
    :type logger: logging.Logger
    :param message: Debug message
    :type message: str
    """
    logger.debug(f"[dim cyan]🔍 {message}[/dim cyan]")

def log_info(logger: logging.Logger, message: str, icon: str = "ℹ️"):
    """Log an info message with styling and optional icon.
    
    :param logger: Logger instance to use
    :type logger: logging.Logger
    :param message: Info message
    :type message: str
    :param icon: Icon to display, defaults to "ℹ️"
    :type icon: str
    """
    logger.info(f"[cyan]{icon} {message}[/cyan]")
