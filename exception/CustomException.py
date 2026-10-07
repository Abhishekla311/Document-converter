import traceback
import sys
class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_details)
    @staticmethod
    def get_detailed_error_message(error_message, error_details:sys):
        _, _, exc_tb = traceback.sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error_in {file_name} , line_numer:{line_number} Error_message:{error_message} , Error_details:{error_details}"

    def __str__(self):
        return self.error_message
    
