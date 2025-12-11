# src/utils/emoji_log.py
"""Emoji-enhanced logging utilities for notebooks and scripts."""

from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)


def info(message: str):
    """💬 Informational message."""
    print(f"{Fore.CYAN}💬 {message}{Style.RESET_ALL}")


def success(message: str):
    """✅ Success message."""
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")


def warn(message: str):
    """⚠️ Warning message."""
    print(f"{Fore.YELLOW}⚠️ {message}{Style.RESET_ALL}")


def error(message: str):
    """❌ Error message."""
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")


def task(message: str):
    """🚀 Task start or progress."""
    print(f"{Fore.BLUE}🚀 {message}{Style.RESET_ALL}")


def done(message: str):
    """🏁 Task completed."""
    print(f"{Fore.MAGENTA}🏁 {message}{Style.RESET_ALL}")


def data(message: str):
    """📊 Data-related log."""
    print(f"{Fore.LIGHTBLUE_EX}📊 {message}{Style.RESET_ALL}")


def save(message: str):
    """💾 File save operation."""
    print(f"{Fore.LIGHTGREEN_EX}💾 {message}{Style.RESET_ALL}")


def step(number: int, message: str):
    """📍 Step indicator with number."""
    print(f"{Fore.CYAN}📍 Step {number}: {message}{Style.RESET_ALL}")


def file(message: str):
    """📄 File-related operation."""
    print(f"{Fore.MAGENTA}📄 {message}{Style.RESET_ALL}")


def debug(message: str):
    """🐛 Debug message."""
    print(f"{Fore.LIGHTBLACK_EX}🐛 {message}{Style.RESET_ALL}")
