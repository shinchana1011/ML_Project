import sys
from src.logger import logging
#this will be common in all project
def error_msg_details(err,err_detail:sys):
    _,_,exc_tb=err_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg=("error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(err))
    )
    return error_msg

class Customexception(Exception):
    def __init__(self,error_msg,err_detail:sys):
        super().__init__(error_msg)
        self.error_msg=error_msg_details(error_msg,err_detail)
    def __str__(self):
        return self.error_msg

# if __name__ == "__main__":
#     try: 
#         a=1/0
#     except Exception as e:
#         logging.info("divide by zerod")
#         raise Customexception(e,sys)
