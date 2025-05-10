import json
from flask import Flask, jsonify, request, Response,stream_with_context
from threading import Thread
from models import db, User
from flask_cors import CORS
import time
import 数据预处理
import 数据清洗
import 数据集划分
import lstm模型训练
import test
import 联邦学习训练
import traceback
import base64
import os

# 创建 Flask 应用
app = Flask(__name__)

# 允许跨域
CORS(app)

# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
db.init_app(app)

@app.route('/')
def home():
    return jsonify({"message": "后端运行成功！"})

# 注册接口
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': '用户名和密码不能为空！'}), 400

    # 检查用户名是否已经存在
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'message': '用户名已存在！'}), 400

    # 创建新用户并保存到数据库
    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': '注册成功！'})

# 登录接口
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username, password=password).first()

    if user:
        return jsonify({'message': '登录成功！', 'user_id': user.id})
    else:
        return jsonify({'message': '用户名或密码错误！'}), 401

#数据预处理接口
# 全局状态存储进度
preprocess_status = {
    "running": False,
    "current": 0,
    "total": 0,
    "message": ""
}

@app.route('/preprocess', methods=['POST'])
def start_preprocess():
    """启动预处理任务（POST）"""
    global preprocess_status
    data = request.json
    data_root = data.get("data_root")
    save_dir = data.get("save_dir")

    # 参数校验
    if not data_root or not save_dir:
        return jsonify({"error": "参数缺失"}), 400
    if not os.path.exists(data_root):
        return jsonify({"error": f"数据目录 {data_root} 不存在"}), 400
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 自动创建保存目录

    # 初始化状态
    preprocess_status.update({
        "running": True,
        "current": 0,
        "total": 0,
        "message": "开始处理..."
    })

    # 启动异步线程处理任务
    thread = Thread(target=run_preprocess, args=(data_root, save_dir))
    thread.start()

    return jsonify({"message": "任务已启动"}), 202

@app.route('/preprogress')
def get_progress():
    """获取进度（SSE）"""
    def generate():
        try:
            while True:
                # 推送当前状态
                data = json.dumps({
                    "running": preprocess_status["running"],
                    "current": preprocess_status["current"],
                    "total": preprocess_status["total"],
                    "message": preprocess_status["message"]
                })
                yield f"data: {data}\n\n"
                time.sleep(0.5)

                # 任务完成后发送特殊事件并退出
                if not preprocess_status["running"]:
                    yield "event: done\ndata: {\"status\": \"complete\"}\n\n"
                    break
        except GeneratorExit:
            print("客户端主动关闭连接")
        except Exception as e:
            print(f"SSE推送异常: {str(e)}")

    return Response(stream_with_context(generate()), content_type="text/event-stream")

def run_preprocess(data_root, save_dir):
    """实际处理函数"""
    global preprocess_status
    try:
        # 修改后的数据处理函数（需支持回调）
        def progress_callback(current, total):
            preprocess_status["current"] = current
            preprocess_status["total"] = total
            preprocess_status["message"] = f"处理中 {current}/{total}"

        # 调用数据处理逻辑
        processed_file_path, final_df = 数据预处理.datapreprocessing(
            data_root, save_dir,
            progress_callback=progress_callback
        )

        # 更新完成状态
        preprocess_status.update({
            "message": f"完成！共处理 {len(final_df)} 条轨迹点",
            "running": False,
            "current": preprocess_status["total"],  # 确保current=total
        })
        time.sleep(1)  # 留出时间让前端接收最终状态
    except Exception as e:
        preprocess_status.update({
            "message": f"错误: {str(e)}",
            "running": False
        })

#数据清洗与可视化接口
# 数据清洗与可视化接口（直接调用封装函数）
clean_status = {
    "running": False,
    "original": 0,
    "cleaned": 0,
    "downsampled": 0,
    "geo_plot": "",
    "time_plot": "",
    "message": ""
}


@app.route('/clean', methods=['POST'])
def start_clean():
    global clean_status

    if clean_status["running"]:
        return jsonify({"error": "已有任务在进行中"}), 400

    data = request.json
    input_path = data.get("input_path")
    # 校验输入路径是否存在且为文件
    if not os.path.isfile(input_path):
        return jsonify({"error": f"输入文件 {input_path} 不存在或不是文件"}), 400
    output_dir = data.get("output_dir", "./clean_output/")  # 默认改为目录形式

    # 参数校验
    if not input_path:
        return jsonify({"error": "需要提供输入路径"}), 400
    if not os.path.exists(input_path):
        return jsonify({"error": f"输入文件 {input_path} 不存在"}), 400

    # 规范化输出目录（确保以斜杠结尾）
    output_dir = os.path.normpath(output_dir) + os.sep  # 自动添加系统分隔符
    os.makedirs(output_dir, exist_ok=True)  # 直接创建目录

    output_path = output_dir  # 就是目录

    # 初始化状态
    clean_status.update({
        "running": True,
        "original": 0,
        "cleaned": 0,
        "downsampled": 0,
        "geo_plot": "",
        "time_plot": "",
        "message": f"结果将保存至：{output_path}   "
    })

    # 启动任务
    Thread(target=run_clean_task, args=(input_path, output_path)).start()
    return jsonify({"message": "数据清洗任务已启动"}), 202


def run_clean_task(input_path, output_base):
    global clean_status
    try:
        # 调用清洗函数
        result = 数据清洗.data_clean(
            input_path=input_path,
            output_path=output_base  # 不带扩展名的路径
        )

        # 处理异常返回值
        if "error" in result:
            raise Exception(result["error"])

        # 更新状态
        clean_status.update({
            "running": False,
            "original": result["original_count"],
            "cleaned": result["cleaned_count"],
            "downsampled": result["downsampled_count"],
            "geo_plot": result["geo_plot"],
            "time_plot": result["time_plot"],
            "message": f"处理完成：保存至 {output_base}"
        })

    except Exception as e:
        clean_status.update({
            "running": False,
            "message": f"处理失败: {str(e)}"
        })
        # 清理可能产生的半成品文件
        for name in ["processed_geolife_cleaned.csv", "processed_geolife_cleaned.pkl"]:
            file_path = os.path.join(output_base, name)
            if os.path.exists(file_path):
                os.remove(file_path)


@app.route('/clean-status')
def get_clean_status():
    def generate():
        while True:
            data = json.dumps({
                "status": clean_status,
                "timestamp": time.time()
            })
            yield f"data: {data}\n\n"
            time.sleep(0.5)
            if not clean_status["running"]:
                # 送一次 done 事件，把所有结果一次性发出去
                done_data = json.dumps(clean_status)
                yield f"event: done\ndata: {done_data}\n\n"
                break
    return Response(stream_with_context(generate()), content_type="text/event-stream")

#数据集划分接口
@app.route('/process-data', methods=['POST'])
def handle_processing():
    data = request.json
    original_dir = data['originalDir']
    save_dir = data['saveDir']

    success, message = 数据集划分.process_trajectory_data(original_dir, save_dir)
    return jsonify({
        'success': success,
        'message': message
    })


#lstm模型训练接口
@app.route('/train-lstm', methods=['POST'])
def start_lstm_training():
    data_dir = request.json['data_dir']

    def generate():
        try:
            generator = lstm模型训练.train_lstm_model(data_dir)
            for event in generator:
                if event['type'] == 'progress':
                    progress_data = {
                        'status': 'progress',
                        'epoch': event['epoch'],
                        'total': event['total_epochs'],
                        'train_loss': event['train_loss'],
                        'val_loss': event['val_loss']
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"
                elif event['type'] == 'complete':
                    complete_data = {
                        'status': 'complete',
                        'test_loss': event['test_loss']
                    }
                    yield f"data: {json.dumps(complete_data)}\n\n"
        except Exception as e:
            error_data = {
                'status': 'error',
                'message': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


#结果可视化接口
@app.route('/api/visualize', methods=['POST'])
def handle_visualization():
    try:
        data = request.json
        if 'model_dir' not in data:
            return jsonify({"error": "缺少模型目录参数"}), 400

        result = test.visualize_results(data['model_dir'])
        if 'error' in result:
            return jsonify(result), 500

        return jsonify({
            "status": "success",
            "mse": result["mse"],
            "mae": result["mae"],
            "rmse": result["rmse"],
            "plot": result["plot"]
        })
    except Exception as e:
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500


# 联邦学习训练接口（支持SSE）
@app.route('/federated/stream', methods=['POST'])
def federated_stream():
    data = request.json
    data_root = data.get('data_root')
    num_clients = int(data.get('num_clients', 5))

    def generate():
        # 创建带缓冲区的生成器

        result_generator = 联邦学习训练.federated_train(data_root, num_clients, as_stream=True)

        try:
            for message in result_generator:
                yield f"data: {json.dumps(message)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

# 初始化数据库
with app.app_context():
    db.create_all()  # 创建所有表

if __name__ == '__main__':
    CORS(app)
    app.run(host='0.0.0.0', port=5000, debug=True)