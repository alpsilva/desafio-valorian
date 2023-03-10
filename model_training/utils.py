from datetime import datetime

def get_now_string():
    return datetime.now().strftime("%d%m%Y_%H%M%S")