// 注册
function registerUser() {
    const u = document.getElementById('register-username').value;
    const p = document.getElementById('register-password').value;
    fetch('http://127.0.0.1:5000/register', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({username: u, password: p})
    })
        .then(r => r.json())
        .then(d => {
            alert(d.message);
            if (d.message === '注册成功！') window.location.href = '../html/login.html';
        })
        .catch(() => alert('注册接口调用失败'));
}

// 登录
function loginUser() {
    const u = document.getElementById('login-username').value;
    const p = document.getElementById('login-password').value;
    fetch('http://127.0.0.1:5000/login', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({username: u, password: p})
    })
        .then(r => {
            if (!r.ok) throw new Error();
            return r.json();
        })
        .then(d => {
            alert(d.message);
            if (d.message === '登录成功！') window.location.href = '../html/dashboard.html';
        })
        .catch(() => alert('用户名或密码错误'));
}

// 页面切换控制器
document.querySelectorAll('.sidebar li').forEach(li => {
    li.addEventListener('click', function () {
        // 切换菜单激活状态
        document.querySelectorAll('.sidebar li').forEach(item =>
            item.classList.remove('active'));
        this.classList.add('active');

        // 切换内容区块
        const sectionId = this.dataset.section + '-section';
        document.querySelectorAll('.content-section').forEach(section =>
            section.classList.remove('active'));
        document.getElementById(sectionId).classList.add('active');
    });
});

// 数据预处理接口
function startPreprocessing() {
    const dataRoot = document.getElementById('data-root').value;
    const saveDir = document.getElementById('save-dir').value;
    let eventSource = null;

    // 1. 启动预处理任务
    fetch('http://127.0.0.1:5000/preprocess', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({data_root: dataRoot, save_dir: saveDir})
    })
        .then(response => {
            if (!response.ok) throw new Error(`启动失败: ${response.status}`);

            // 2. 连接SSE进度通道
            eventSource = new EventSource('http://127.0.0.1:5000/preprogress');

            // 处理实时进度更新
            eventSource.addEventListener('message', event => {
                const status = JSON.parse(event.data);
                updateProgress(status);
            });

            // 处理任务完成事件（后端需要发送event: done）
            eventSource.addEventListener('done', event => {
                const result = JSON.parse(event.data);
                console.log('任务完成:', result);
                eventSource.close();
                completeProgress();
            });

            // 处理错误（区分正常关闭和异常中断）
            eventSource.onerror = () => {
                // readyState: 0=CONNECTING, 1=OPEN, 2=CLOSED
                if (eventSource.readyState === EventSource.CLOSED) {
                    console.log('连接正常关闭');
                } else {
                    handleError(new Error('实时进度连接异常中断'));
                }
                eventSource.close();
            };
        })
        .catch(error => {
            handleError(error);
            if (eventSource) eventSource.close();
        });

    // 进度更新函数
    function updateProgress(status) {
        document.getElementById('status').textContent = status.message;

        // 更新进度条
        if (status.current > 0 && status.total > 0) {
            const percent = Math.min(100, (status.current / status.total) * 100);
            document.getElementById('progress-bar').style.width = `${percent}%`;

            // 如果进度接近完成时显示缓冲状态
            if (percent >= 99.5 && percent < 100) {
                document.getElementById('status').textContent = "正在完成最后的处理...";
            }
        }
    }

    // 任务完成函数
    function completeProgress() {
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('status').textContent = "数据处理完成！";
        document.getElementById('progress-bar').classList.add('completed'); // 添加完成样式
    }

    // 统一错误处理
    function handleError(error) {
        console.error('Error:', error);
        document.getElementById('status').textContent = `错误: ${error.message}`;
        document.getElementById('progress-bar').style.backgroundColor = '#ff4444'; // 错误颜色
        if (eventSource) eventSource.close();
    }
}

// 数据清洗功能（SSE + 最终渲染）
let cleaningEventSource = null;

function setupCleanSSE() {
  // 关闭上次的连接
  if (cleaningEventSource) cleaningEventSource.close();

  cleaningEventSource = new EventSource('http://127.0.0.1:5000/clean-status');

  cleaningEventSource.addEventListener('message', e => {
    const payload = JSON.parse(e.data);
    const s = payload.status;
    // 仅在处理中持续更新（这时通常是 0）
    document.getElementById('raw-count').textContent       = s.original;
    document.getElementById('clean-count').textContent     = s.cleaned;
    document.getElementById('downsample-count').textContent= s.downsampled;
  });

  cleaningEventSource.addEventListener('done', e => {
    const s = JSON.parse(e.data);
    // **在 done 时再把最终统计量写一遍**
    document.getElementById('raw-count').textContent       = s.original;
    document.getElementById('clean-count').textContent     = s.cleaned;
    document.getElementById('downsample-count').textContent= s.downsampled;

    // 渲染两张 Base64 图，并加上自适应样式
    document.getElementById('clean-vis-container').innerHTML = `
      <h4>地理分布：</h4>
      <div class="visualization-container">
        <img src="data:image/png;base64,${s.geo_plot}"
             class="visualization-img"
             alt="地理分布图"
             style="max-width:100%; height:auto;" />
      </div>
      <h4>时间分布：</h4>
      <div class="visualization-container">
        <img src="data:image/png;base64,${s.time_plot}"
             class="visualization-img"
             alt="时间分布图"
             style="max-width:100%; height:auto;" />
      </div>
    `;
    cleaningEventSource.close();
  });

  cleaningEventSource.onerror = () => {
    cleaningEventSource.close();
    document.getElementById('clean-vis-container').innerHTML =
      '<div class="error-message">连接中断</div>';
  };
}

function startCleaning() {
  const rawDir      = document.getElementById('raw-data-dir').value;
  const saveDir     = document.getElementById('clean-save-dir').value;
  const visContainer= document.getElementById('clean-vis-container');

  // 重置界面
  visContainer.innerHTML = '<div class="loading-spinner"></div>';
  ['raw-count','clean-count','downsample-count']
    .forEach(id => document.getElementById(id).textContent = '0');

  fetch('http://127.0.0.1:5000/clean', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({input_path: rawDir, output_dir: saveDir})
  })
  .then(resp => {
    if (resp.status !== 202) {
      return resp.json().then(err=>{ throw new Error(err.error || '启动失败') });
    }
    setupCleanSSE();
  })
  .catch(err => {
    visContainer.innerHTML = `<div class="error-message">${err.message}</div>`;
  });
}

// 暴露函数给 HTML onclick 使用
window.startCleaning = startCleaning;

//数据集划分接口
async function startDivision() {
    const originalDir = document.getElementById('originalDir').value;
    const saveDir = document.getElementById('saveDir').value;
    const statusElement = document.getElementById('division-status');

    try {
        statusElement.textContent = "处理中，请稍候...";
        statusElement.className = "status-message processing";

        const response = await fetch('http://127.0.0.1:5000/process-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ originalDir, saveDir })
        });

        const result = await response.json();
        if (result.success) {
            statusElement.textContent = result.message || "处理完成！";
            statusElement.className = "status-message success";
            showAlert('success', result.message);
        } else {
            statusElement.textContent = `处理失败: ${result.message}`;
            statusElement.className = "status-message error";
            showAlert('error', `处理失败: ${result.message}`);
        }
    } catch (error) {
        statusElement.textContent = `网络错误: ${error.message}`;
        statusElement.className = "status-message error";
        showAlert('error', `网络错误: ${error.message}`);
    }
}

// 示例显示函数
function showAlert(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert ${type}`;
    alertDiv.textContent = message;
    document.body.prepend(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}

// lstm模型训练接口
function startLSTMTraining() {
    const dataDir = document.getElementById('lstm-input').value;
    if (!dataDir) return alert('请先输入训练数据目录');

    // 重置显示状态
    document.getElementById('epoch-progress').textContent = '0/0';
    document.getElementById('training-progress').style.width = '0%';
    document.getElementById('train-loss').textContent = '0.000000';
    document.getElementById('val-loss').textContent = '0.000000';
    document.getElementById('test-loss-value').textContent = '0.000000';
    document.getElementById('lstm-status').textContent = '训练启动中...';

    // 启动训练
    fetch('http://127.0.0.1:5000/train-lstm', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ data_dir: dataDir })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`请求失败，状态码：${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        const processChunk = ({ done, value }) => {
            if (done) {
                console.log('训练流结束');
                return;
            }

            const chunk = decoder.decode(value, { stream: true });
            // 处理可能分多次接收的事件
            const events = chunk.split('\n\n');
            events.forEach(eventStr => {
                if (eventStr.trim() === '') return;
                const dataLine = eventStr.split('\n').find(line => line.startsWith('data: '));
                if (dataLine) {
                    const data = JSON.parse(dataLine.substring(6)); // 去掉'data: '
                    handleTrainingEvent(data);
                }
            });

            // 继续读取下一个数据块
            reader.read().then(processChunk);
        };

        reader.read().then(processChunk).catch(error => {
            console.error('读取流失败:', error);
            document.getElementById('lstm-status').textContent = '连接异常中断';
        });
    })
    .catch(error => {
        console.error('启动训练失败:', error);
        document.getElementById('lstm-status').textContent = '训练启动失败: ' + error.message;
    });
}

function handleTrainingEvent(data) {
    switch(data.status) {
        case 'progress':
            document.getElementById('epoch-progress').textContent =
                `${data.epoch}/${data.total}`;
            document.getElementById('training-progress').style.width =
                `${(data.epoch / data.total) * 100}%`;
            document.getElementById('train-loss').textContent =
                data.train_loss.toFixed(6);
            document.getElementById('val-loss').textContent =
                data.val_loss.toFixed(6);
            break;
        case 'complete':
            document.getElementById('test-loss-value').textContent =
                data.test_loss.toFixed(6);
            document.getElementById('lstm-status').textContent = '训练完成！';
            break;
        case 'error':
            document.getElementById('lstm-status').textContent =
                `错误: ${data.message}`;
            break;
    }
}


//结果可视化接口
function generateResultVisualization() {
    const modelDir = document.getElementById('model-dir').value;
    if (!modelDir) {
        alert('请输入模型目录路径');
        return;
    }

    const statusElement = document.getElementById('lstm-status');
    const plotElement = document.getElementById('result-plot');
    const mseElement = document.getElementById('mse-value');
    const maeElement = document.getElementById('mae-value');
    const rmseElement = document.getElementById('rmse-value');
    const loadingSpinner = document.getElementById('plot-loading');

    // 重置显示状态
    plotElement.style.display = 'none';
    loadingSpinner.style.display = 'block';
    statusElement.textContent = '正在生成可视化结果...';

    // 禁用按钮防止重复提交
    const btn = document.querySelector('#visualization-section .btn-primary');
    btn.disabled = true;

    fetch('http://127.0.0.1:5000/api/visualize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_dir: modelDir })
    })
    .then(response => {
        if (!response.ok) throw new Error(`HTTP错误 ${response.status}`);
        return response.json();
    })
    .then(data => {
        if (data.error) throw new Error(data.error);

        // 更新数据
        mseElement.textContent = data.mse.toFixed(6);
        maeElement.textContent = data.mae.toFixed(6);
        rmseElement.textContent = data.rmse.toFixed(6);
        plotElement.src = `data:image/png;base64,${data.plot}`;
        plotElement.style.display = 'block';
        statusElement.textContent = '可视化生成成功';
    })
    .catch(error => {
        console.error('Error:', error);
        statusElement.textContent = '生成失败'  + error.message;
        alert(`错误: ${error.message}`);
    })
    .finally(() => {
        loadingSpinner.style.display = 'none';
        btn.disabled = false;
    });
}

// 联邦学习训练功能实现
let federatedEventSource = null;

function startFederatedTraining() {
    const dataDir = document.getElementById('federated-data-dir').value;
    const numClients = document.getElementById('num-clients').value;

    // 重置状态
    document.getElementById('client-training-status').innerHTML = '';
    document.getElementById('loss-plot').src = '';
    document.querySelectorAll('.metric-value').forEach(v => v.textContent = '0.000000');

    // 禁用按钮
    const btn = document.querySelector('#federated-section button');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 训练中...';

    // 创建新的EventSource连接
    if (federatedEventSource) federatedEventSource.close();

    fetch('http://127.0.0.1:5000/federated/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            data_root: dataDir,
            num_clients: parseInt(numClients)
        })
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        const processChunk = ({ done, value }) => {
            if (done) {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-play"></i> 开始联邦训练';
                return;
            }

            // 处理多事件情况
            const chunk = decoder.decode(value);
            chunk.split('\n\n').forEach(eventStr => {
                if (!eventStr.trim()) return;

                try {
                    const eventData = eventStr.replace('data: ', '');
                    const event = JSON.parse(eventData);
                    handleFederatedEvent(event);
                } catch (e) {
                    console.error('解析错误:', e);
                }
            });

            // 继续读取
            return reader.read().then(processChunk);
        };

        return reader.read().then(processChunk);
    })
    .catch(error => {
        console.error('请求失败:', error);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> 开始联邦训练';
        alert('联邦训练启动失败: ' + error.message);
    });
}

// 事件处理器
function handleFederatedEvent(event) {
    switch(event.type) {
        case 'round_start':
            updateRoundProgress(event.round, event.total_rounds);
            break;

        case 'client_start':
            createClientEntry(event.client, event.samples);
            break;

        case 'debug_info':
            updateClientDimensions(event.client, event.input_shape, event.label_shape);
            break;

        case 'client_progress':
            updateClientTrainingStatus(event.client, event.epoch, event.loss);
            break;

        case 'validation':
            updateGlobalMetrics(event);
            break;

        case 'final_result':
            showTrainingResult(event.result);
            break;
    }
}

function updateRoundProgress(currentRound, totalRounds) {
    // 更新进度条
    const progressPercent = (currentRound / totalRounds) * 100;
    document.getElementById('federated-progress').style.width = `${progressPercent}%`;

    // 更新轮次显示
    document.getElementById('round-progress').textContent =
        `${currentRound}/${totalRounds}`;
}
// 创建客户端状态条目
function createClientEntry(clientName, samples) {
    const container = document.getElementById('client-training-status');

    // 先检查是否已存在该客户端的卡片
    const existingCard = [...document.querySelectorAll('.client-card')].find(
        card => card.querySelector('.client-name').textContent === clientName
    );

    if (existingCard) {
        // 如果卡片已存在，仅更新样本数
        existingCard.querySelector('.training-status div:nth-child(2)').textContent =
            `样本数：${samples}`;
        return; // 不再创建新卡片
    }

    const clientDiv = document.createElement('div');
    clientDiv.className = 'client-card';
    clientDiv.innerHTML = `
        <div class="client-header">
            <span class="client-name">${clientName}</span>
            <span class="status-indicator1"></span>
        </div>
        <div class="client-body">
            <div class="data-dimensions">
                <span>输入维度：加载中...</span>
                <span>标签维度：加载中...</span>
            </div>
            <div class="training-status">
                <div>轮次：<span class="round">0</span></div>
                <div>样本数：${samples}</div>
                <div>训练损失：<span class="loss">0.000000</span></div>
            </div>
        </div>
    `;

    container.appendChild(clientDiv);
}

// 更新客户端维度信息
function updateClientDimensions(clientName, inputShape, labelShape) {
    const clientDiv = [...document.querySelectorAll('.client-card')]
        .find(card => card.querySelector('.client-name').textContent === clientName);

    if (clientDiv) {
        const dimSpan = clientDiv.querySelector('.data-dimensions');
        dimSpan.innerHTML = `
            <span>输入维度：${inputShape.join('×')}</span>
            <span>标签维度：${labelShape.join('×')}</span>
        `;
    }
}

// 更新客户端训练状态
function updateClientTrainingStatus(clientName, epoch, loss) {
    const clientDiv = [...document.querySelectorAll('.client-card')]
        .find(card => card.querySelector('.client-name').textContent === clientName);

    if (clientDiv) {
        clientDiv.querySelector('.round').textContent = epoch;
        clientDiv.querySelector('.loss').textContent = loss.toFixed(6);
        clientDiv.querySelector('.status-indicator1').style.backgroundColor = '#4CAF50';
    }
}

// 更新全局验证指标
function updateGlobalMetrics(event) {
    // 更新指标显示
    document.getElementById('current-mse').textContent = event.mse.toFixed(6);
    document.getElementById('current-mae').textContent = event.mae.toFixed(6);
    document.getElementById('current-r2').textContent = event.r2.toFixed(6);

    // 更新进度条
    const progress = (event.round / event.total_rounds) * 100;
    document.getElementById('federated-progress').style.width = `${progress}%`;
    document.getElementById('round-progress').textContent = `${event.round}/${event.total_rounds}`;
}

// 显示最终训练结果
function showTrainingResult(result) {
    // 显示损失曲线
    const plotImg = document.getElementById('loss-plot');
    plotImg.src = `data:image/png;base64,${result.loss_plot}`;

    // 更新最终指标
    const lastMetrics = result.val_metrics.slice(-1)[0];
    document.getElementById('current-mse').textContent = lastMetrics.mse.toFixed(6);
    document.getElementById('current-mae').textContent = lastMetrics.mae.toFixed(6);
    document.getElementById('current-r2').textContent = lastMetrics.r2.toFixed(6);
}