# #注册请求
# import requests
#
# url = 'http://127.0.0.1:5000/register'
# data = {
#     'username': 'testuser',
#     'password': 'password123'
# }
#
# response = requests.post(url, json=data)
#
# print(response.json())  # 打印返回结果

#登录请求
import requests

url = 'http://127.0.0.1:5000/login'
data = {
    'username': 'testuser',
    'password': 'password123'
}

response = requests.post(url, json=data)

print(response.json())  # 打印返回结果
