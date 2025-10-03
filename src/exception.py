import sys
import traceback
from typing import Optional

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(error_message)
        
        if error_detail is None:
            error_detail = sys.exc_info()
        
        self.error_message = self._get_detailed_error_message(error_message, error_detail)

    def _get_detailed_error_message(self, error_message: str, error_detail: tuple) -> str:
        """
        Create detailed error message with file, line number, and error details
        """
        _, _, exc_tb = error_detail
        
        if exc_tb is not None:
            filename = exc_tb.tb_frame.f_code.co_filename.split("/")[-1]
            line_number = exc_tb.tb_lineno
            function_name = exc_tb.tb_frame.f_code.co_name
            
            error_msg = (
                f"Error occurred in file: [{filename}] "
                f"at line: [{line_number}] "
                f"in function: [{function_name}] "
                f"Error message: [{error_message}]"
            )
        else:
            error_msg = f"Error message: [{error_message}]"
        
        return error_msg

    def __str__(self):
        return self.error_message