import hashlib
import datetime

def generate_daily_password():
    today = datetime.date.today().isoformat()
    hash_obj = hashlib.md5(today.encode()).hexdigest()
    return hash_obj[:4]

def verify_password(password):
    return password == generate_daily_password()

def get_today_password():
    return generate_daily_password()

if __name__ == '__main__':
    print(f"今日密码: {generate_daily_password()}")
    print(f"日期: {datetime.date.today().isoformat()}")
