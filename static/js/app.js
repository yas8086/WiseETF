let equityChart = null;

document.addEventListener('DOMContentLoaded', function() {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('end-date').value = today;
    
    document.getElementById('backtest-form').addEventListener('submit', function(e) {
        e.preventDefault();
        runBacktest();
    });
    
    document.getElementById('strategy-type').addEventListener('change', updateStrategyParams);
    
    updateStrategyParams();
});

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    document.getElementById(tabName + '-tab').style.display = 'block';
    
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    event.target.classList.add('active');
}

function updateStrategyParams() {
    const strategyType = document.getElementById('strategy-type').value;
    const paramsDiv = document.getElementById('strategy-params');
    
    if (strategyType === 'momentum') {
        paramsDiv.innerHTML = `
            <div class="col-md-6">
                <label class="form-label">动量周期 (天)</label>
                <input type="number" class="form-control" id="lookback-period" value="20">
            </div>
            <div class="col-md-6">
                <label class="form-label">持有数量</label>
                <select class="form-select" id="top-n">
                    <option value="1" selected>1只</option>
                    <option value="2">2只</option>
                    <option value="3">3只</option>
                </select>
            </div>
        `;
    } else if (strategyType === 'ma') {
        paramsDiv.innerHTML = `
            <div class="col-md-6">
                <label class="form-label">短期均线</label>
                <select class="form-select" id="short-ma">
                    <option value="5">MA5</option>
                    <option value="10" selected>MA10</option>
                    <option value="20">MA20</option>
                </select>
            </div>
            <div class="col-md-6">
                <label class="form-label">长期均线</label>
                <select class="form-select" id="long-ma">
                    <option value="50" selected>MA50</option>
                    <option value="100">MA100</option>
                    <option value="200">MA200</option>
                </select>
            </div>
        `;
    } else if (strategyType === 'dual_momentum') {
        paramsDiv.innerHTML = `
            <div class="col-md-4">
                <label class="form-label">动量周期 (天)</label>
                <input type="number" class="form-control" id="lookback-period" value="20">
            </div>
            <div class="col-md-4">
                <label class="form-label">短期均线</label>
                <select class="form-select" id="ma-short">
                    <option value="10" selected>MA10</option>
                    <option value="20">MA20</option>
                </select>
            </div>
            <div class="col-md-4">
                <label class="form-label">长期均线</label>
                <select class="form-select" id="ma-long">
                    <option value="50" selected>MA50</option>
                    <option value="100">MA100</option>
                </select>
            </div>
        `;
    }
}

async function runBacktest() {
    showLoading('正在运行回测...');
    
    const etfCodes = Array.from(document.getElementById('etf-codes').selectedOptions).map(opt => opt.value);
    const strategyType = document.getElementById('strategy-type').value;
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const initialCapital = document.getElementById('initial-capital').value;
    
    let params = {};
    if (strategyType === 'momentum') {
        params = {
            lookback_period: parseInt(document.getElementById('lookback-period').value),
            top_n: parseInt(document.getElementById('top-n').value)
        };
    } else if (strategyType === 'ma') {
        params = {
            short_ma: parseInt(document.getElementById('short-ma').value),
            long_ma: parseInt(document.getElementById('long-ma').value)
        };
    } else if (strategyType === 'dual_momentum') {
        params = {
            lookback_period: parseInt(document.getElementById('lookback-period').value),
            ma_short: parseInt(document.getElementById('ma-short').value),
            ma_long: parseInt(document.getElementById('ma-long').value)
        };
    }
    
    try {
        const response = await fetch('/api/backtest/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                etf_codes: etfCodes,
                strategy_type: strategyType,
                params: params,
                start_date: startDate,
                end_date: endDate,
                initial_capital: parseFloat(initialCapital)
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.result);
        } else {
            alert('回测失败: ' + data.message);
        }
    } catch (error) {
        alert('请求失败: ' + error.message);
    }
    
    hideLoading();
}

function displayResults(result) {
    document.getElementById('results-section').style.display = 'block';
    
    document.getElementById('total-return').textContent = result.total_return.toFixed(2) + '%';
    document.getElementById('annual-return').textContent = result.annual_return.toFixed(2) + '%';
    document.getElementById('max-drawdown').textContent = result.max_drawdown.toFixed(2) + '%';
    document.getElementById('sharpe-ratio').textContent = result.sharpe_ratio.toFixed(3);
    document.getElementById('win-rate').textContent = result.win_rate.toFixed(2) + '%';
    document.getElementById('trade-count').textContent = result.trade_count;
    
    drawEquityCurve(result.equity_curve);
    
    displaySignals(result.signals.slice(-20));
}

function drawEquityCurve(equityData) {
    const ctx = document.getElementById('equity-chart').getContext('2d');
    
    if (equityChart) {
        equityChart.destroy();
    }
    
    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: equityData.map(d => d.date.split('T')[0]),
            datasets: [{
                label: '账户净值',
                data: equityData.map(d => d.portfolio_value),
                borderColor: 'rgb(13, 110, 253)',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return '净值: ¥' + context.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2});
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '¥' + (value/10000).toFixed(1) + '万';
                        }
                    }
                }
            }
        }
    });
}

function displaySignals(signals) {
    const tbody = document.getElementById('signals-tbody');
    tbody.innerHTML = '';
    
    signals.reverse().forEach(signal => {
        const tr = document.createElement('tr');
        const actionClass = signal.action === 'BUY' ? 'btn-buy' : 'btn-sell';
        tr.innerHTML = `
            <td>${signal.date}</td>
            <td>${signal.etf_code}</td>
            <td class="${actionClass}">${signal.action === 'BUY' ? '买入' : '卖出'}</td>
            <td>${signal.price.toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
    });
}

async function runOptimization() {
    showLoading('正在优化策略，这可能需要几分钟...');
    
    const etfCodes = Array.from(document.getElementById('etf-codes').selectedOptions).map(opt => opt.value);
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    
    try {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                etf_codes: etfCodes,
                start_date: startDate,
                end_date: endDate
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayOptimizeResults(data.results);
        } else {
            alert('优化失败: ' + data.message);
        }
    } catch (error) {
        alert('请求失败: ' + error.message);
    }
    
    hideLoading();
}

function displayOptimizeResults(results) {
    document.getElementById('optimize-results').style.display = 'block';
    const tbody = document.getElementById('optimize-tbody');
    tbody.innerHTML = '';
    
    results.forEach((result, index) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${index + 1}</td>
            <td><strong>${result.strategy_name}</strong></td>
            <td class="text-success"><strong>${result.annual_return.toFixed(2)}%</strong></td>
            <td>${result.total_return.toFixed(2)}%</td>
            <td class="text-danger">${result.max_drawdown.toFixed(2)}%</td>
            <td>${result.sharpe_ratio.toFixed(3)}</td>
            <td>${result.win_rate.toFixed(2)}%</td>
            <td><button class="btn btn-sm btn-outline-primary" onclick="useStrategy(${index})">使用此策略</button></td>
        `;
        tbody.appendChild(tr);
    });
    
    window.optimizeResults = results;
}

function useStrategy(index) {
    const strategy = window.optimizeResults[index];
    alert(`已选择策略: ${strategy.strategy_name}\n年化收益: ${strategy.annual_return}%\n最大回撤: ${strategy.max_drawdown}%`);
    showTab('backtest');
}

async function updateData() {
    showLoading('正在更新ETF数据...');
    
    try {
        const response = await fetch('/api/etf/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert(data.message);
        } else {
            alert('更新失败');
        }
    } catch (error) {
        alert('更新失败: ' + error.message);
    }
    
    hideLoading();
}

function showLoading(text = '正在处理...') {
    document.getElementById('loading-text').textContent = text;
    const modal = new bootstrap.Modal(document.getElementById('loading-modal'));
    modal.show();
}

function hideLoading() {
    const modalEl = document.getElementById('loading-modal');
    const modal = bootstrap.Modal.getInstance(modalEl);
    if (modal) {
        modal.hide();
    }
}
