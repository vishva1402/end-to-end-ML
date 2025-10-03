from src.logger import logger
from src.exception import CustomException
import sys

def divide_numbers(a: float, b: float) -> float:
    """Example function showing how to use logger and exception"""
    try:
        logger.info(f"Starting division: {a} / {b}")
        
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        result = a / b
        logger.info(f"Division successful. Result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error in divide_numbers: {str(e)}")
        raise CustomException(f"Division failed: {str(e)}", sys.exc_info())

# Example usage
if __name__ == "__main__":
    try:
        result = divide_numbers(10, 0)
    except CustomException as e:
        logger.error(f"Custom exception caught: {e}")