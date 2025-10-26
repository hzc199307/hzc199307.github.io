// 交通出行方案计算器模块
export class TransportCalculator {
    constructor() {
        // 交通方案类型
        this.transportTypes = {
            taxi: {
                name: '出租车',
                type: 'taxi',
                color: 'rgba(34, 197, 94, 1)',
                bgColor: 'rgba(34, 197, 94, 0.1)'
            },
            bike: {
                name: '共享单车',
                type: 'bike',
                color: 'rgba(59, 130, 246, 1)',
                bgColor: 'rgba(59, 130, 246, 0.1)'
            },
            subway: {
                name: '地铁',
                type: 'subway',
                color: 'rgba(168, 85, 247, 1)',
                bgColor: 'rgba(168, 85, 247, 0.1)'
            },
            subwayStudent: {
                name: '地铁(学生)',
                type: 'subwayStudent',
                color: 'rgba(236, 72, 153, 1)',
                bgColor: 'rgba(236, 72, 153, 0.1)'
            }
        };
        
        this.selectedPlans = ['taxi', 'bike', 'subway']; // 默认选择的方案
        this.chart = null;
    }

    // 出租车计费：起步价10元（2公里内），超出部分 y = 2.7x + 4.6（x > 2公里）
    calculateTaxiCost(distance) {
        if (distance <= 2) {
            return 10;
        } else {
            // y = 10 + 2.7(x - 2) = 10 + 2.7x - 5.4 = 2.7x + 4.6
            return 2.7 * distance + 4.6;
        }
    }

    // 共享单车计费：匀速10 km/h，每15分钟1.5元 → y = 0.6x（x为公里数）
    calculateBikeCost(distance) {
        // 时间（小时）= 距离 / 速度 = distance / 10
        // 时间（分钟）= distance / 10 * 60 = 6 * distance
        // 费用 = (时间分钟 / 15) * 1.5 = (6 * distance / 15) * 1.5 = 0.6 * distance
        return 0.6 * distance;
    }

    // 地铁计费：分段计价
    // 0-4km: 2元
    // 4-12km: 每4km+1元
    // 12-24km: 每6km+1元
    // >24km: 每8km+1元
    calculateSubwayCost(distance, isStudent = false) {
        let cost = 0;
        
        if (distance <= 4) {
            cost = 2;
        } else if (distance <= 12) {
            // 前4km: 2元，4-12km: 每4km+1元
            cost = 2;
            const extraDistance = distance - 4;
            cost += Math.ceil(extraDistance / 4) * 1;
        } else if (distance <= 24) {
            // 前4km: 2元，4-12km: 2元，12-24km: 每6km+1元
            cost = 2 + Math.ceil(8 / 4) * 1; // 前12km的费用 = 2 + 2 = 4元
            const extraDistance = distance - 12;
            cost += Math.ceil(extraDistance / 6) * 1;
        } else {
            // 前4km: 2元，4-12km: 2元，12-24km: 2元，>24km: 每8km+1元
            cost = 2 + Math.ceil(8 / 4) * 1 + Math.ceil(12 / 6) * 1; // 前24km的费用 = 2 + 2 + 2 = 6元
            const extraDistance = distance - 24;
            cost += Math.ceil(extraDistance / 8) * 1;
        }
        
        // 学生5折
        if (isStudent) {
            cost = cost * 0.5;
        }
        
        return cost;
    }

    // 计算指定方案的费用
    calculateCost(planType, distance) {
        switch(planType) {
            case 'taxi':
                return this.calculateTaxiCost(distance);
            case 'bike':
                return this.calculateBikeCost(distance);
            case 'subway':
                return this.calculateSubwayCost(distance, false);
            case 'subwayStudent':
                return this.calculateSubwayCost(distance, true);
            default:
                return 0;
        }
    }

    // 更新选择的方案
    updateSelectedPlans(plans) {
        this.selectedPlans = plans;
    }

    // 生成对比数据
    generateComparisonData(maxDistance = 30) {
        const data = [];
        const step = maxDistance / 15; // 生成15个数据点

        for (let distance = 0; distance <= maxDistance; distance += step) {
            const row = {
                distance: distance.toFixed(1)
            };

            this.selectedPlans.forEach(plan => {
                const cost = this.calculateCost(plan, distance);
                row[plan] = cost.toFixed(2);
            });

            // 找出最便宜的方案
            let minCost = Infinity;
            let cheapest = '';
            this.selectedPlans.forEach(plan => {
                const cost = parseFloat(row[plan]);
                if (cost < minCost) {
                    minCost = cost;
                    cheapest = this.transportTypes[plan].name;
                }
            });
            row.cheapest = cheapest;

            data.push(row);
        }

        return data;
    }

    // 生成图表数据
    generateChartData(maxDistance = 30) {
        const labels = [];
        const datasets = {};
        const step = 0.5;

        // 初始化数据集
        this.selectedPlans.forEach(plan => {
            datasets[plan] = [];
        });

        for (let distance = 0; distance <= maxDistance; distance += step) {
            labels.push(distance.toFixed(1));
            
            this.selectedPlans.forEach(plan => {
                const cost = this.calculateCost(plan, distance);
                datasets[plan].push(cost.toFixed(2));
            });
        }

        return { labels, datasets };
    }

    // 创建对比图表
    createComparisonChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // 销毁旧图表
        if (this.chart) {
            this.chart.destroy();
        }

        const chartData = this.generateChartData(30);
        
        // 构建数据集
        const chartDatasets = this.selectedPlans.map(plan => {
            const planInfo = this.transportTypes[plan];
            return {
                label: planInfo.name,
                data: chartData.datasets[plan],
                borderColor: planInfo.color,
                backgroundColor: planInfo.bgColor,
                borderWidth: 3,
                tension: 0.1,
                pointRadius: 0,
                pointHoverRadius: 6
            };
        });

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: chartDatasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ¥' + context.parsed.y;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: '距离（公里）',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '费用（元）',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            callback: function(value) {
                                return '¥' + value;
                            }
                        }
                    }
                }
            }
        });
    }

    // 生成计费规则说明HTML
    generatePricingRulesHTML() {
        return `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div class="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 border-2 border-green-300">
                    <div class="flex items-center mb-4">
                        <div class="text-4xl mr-3">🚕</div>
                        <h4 class="text-xl font-bold text-green-800">出租车</h4>
                    </div>
                    <div class="space-y-2 text-gray-700">
                        <p class="font-semibold">分段计费：</p>
                        <p>• 0-2公里：<span class="font-bold text-green-700">10元</span>（起步价）</p>
                        <p>• 超过2公里：<span class="font-bold text-green-700">y = 2.7x + 4.6</span></p>
                        <p class="text-sm text-gray-600 mt-3">即：10 + 2.7(x - 2)</p>
                    </div>
                </div>

                <div class="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border-2 border-blue-300">
                    <div class="flex items-center mb-4">
                        <div class="text-4xl mr-3">🚲</div>
                        <h4 class="text-xl font-bold text-blue-800">共享单车</h4>
                    </div>
                    <div class="space-y-2 text-gray-700">
                        <p class="font-semibold">按时间计费：</p>
                        <p>• 速度：<span class="font-bold text-blue-700">10 km/h</span></p>
                        <p>• 价格：<span class="font-bold text-blue-700">每15分钟1.5元</span></p>
                        <p class="text-sm text-gray-600 mt-3">函数：y = 0.6x（x为公里数）</p>
                    </div>
                </div>

                <div class="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 border-2 border-purple-300">
                    <div class="flex items-center mb-4">
                        <div class="text-4xl mr-3">🚇</div>
                        <h4 class="text-xl font-bold text-purple-800">地铁</h4>
                    </div>
                    <div class="space-y-2 text-gray-700 text-sm">
                        <p class="font-semibold">分段计价：</p>
                        <p>• 0-4km：<span class="font-bold text-purple-700">2元</span></p>
                        <p>• 4-12km：<span class="font-bold text-purple-700">每4km+1元</span></p>
                        <p>• 12-24km：<span class="font-bold text-purple-700">每6km+1元</span></p>
                        <p>• >24km：<span class="font-bold text-purple-700">每8km+1元</span></p>
                        <p class="text-sm text-pink-600 mt-2">🎓 学生享受5折优惠</p>
                    </div>
                </div>
            </div>
        `;
    }

    // 生成详细分析HTML
    generateDetailedAnalysis() {
        const distances = [5, 10, 15, 20, 25, 30];
        let html = `
            <div class="bg-white rounded-xl p-6 mb-6">
                <h4 class="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                    <span class="text-3xl mr-3">📊</span>
                    不同距离费用对比分析
                </h4>
                <div class="overflow-x-auto">
                    <table class="data-table w-full">
                        <thead>
                            <tr>
                                <th class="text-center">距离(km)</th>
                                ${this.selectedPlans.map(plan => 
                                    `<th class="text-center">${this.transportTypes[plan].name}</th>`
                                ).join('')}
                                <th class="text-center">最优选择</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

        distances.forEach(distance => {
            const costs = {};
            let minCost = Infinity;
            let bestPlan = '';

            this.selectedPlans.forEach(plan => {
                costs[plan] = this.calculateCost(plan, distance);
                if (costs[plan] < minCost) {
                    minCost = costs[plan];
                    bestPlan = plan;
                }
            });

            html += `<tr>`;
            html += `<td class="text-center font-bold">${distance}</td>`;
            
            this.selectedPlans.forEach(plan => {
                const isBest = plan === bestPlan;
                const className = isBest ? 'bg-yellow-100 font-bold text-green-700' : '';
                html += `<td class="text-center ${className}">¥${costs[plan].toFixed(2)}</td>`;
            });

            html += `<td class="text-center font-bold text-indigo-700">${this.transportTypes[bestPlan].name}</td>`;
            html += `</tr>`;
        });

        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        return html;
    }

    // 生成决策建议HTML
    generateDecisionAdvice() {
        return `
            <div class="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 border-2 border-indigo-300">
                <div class="flex items-start">
                    <div class="text-5xl mr-4">💡</div>
                    <div class="flex-1">
                        <h4 class="text-2xl font-bold text-indigo-800 mb-4">智能出行建议</h4>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                            <div class="bg-white rounded-lg p-4">
                                <h5 class="font-bold text-green-800 mb-3 flex items-center">
                                    <span class="text-2xl mr-2">🚕</span>
                                    出租车适用场景
                                </h5>
                                <ul class="space-y-2 text-gray-700 text-sm">
                                    <li>✓ 短途出行（2公里内最划算）</li>
                                    <li>✓ 携带大件行李</li>
                                    <li>✓ 赶时间或恶劣天气</li>
                                    <li>✓ 多人拼车分摊费用</li>
                                    <li class="text-red-600">✗ 长途出行费用较高</li>
                                </ul>
                            </div>

                            <div class="bg-white rounded-lg p-4">
                                <h5 class="font-bold text-blue-800 mb-3 flex items-center">
                                    <span class="text-2xl mr-2">🚲</span>
                                    共享单车适用场景
                                </h5>
                                <ul class="space-y-2 text-gray-700 text-sm">
                                    <li>✓ 短中途出行（5-15公里）</li>
                                    <li>✓ 费用最经济实惠</li>
                                    <li>✓ 锻炼身体，环保出行</li>
                                    <li>✓ 避开交通拥堵</li>
                                    <li class="text-red-600">✗ 体力消耗大，速度较慢</li>
                                </ul>
                            </div>

                            <div class="bg-white rounded-lg p-4">
                                <h5 class="font-bold text-purple-800 mb-3 flex items-center">
                                    <span class="text-2xl mr-2">🚇</span>
                                    地铁适用场景
                                </h5>
                                <ul class="space-y-2 text-gray-700 text-sm">
                                    <li>✓ 中长途出行（10公里以上）</li>
                                    <li>✓ 准点准时，不受路况影响</li>
                                    <li>✓ 舒适安全，可以休息</li>
                                    <li>✓ 学生优惠力度大（5折）</li>
                                    <li class="text-red-600">✗ 需要步行到站点</li>
                                </ul>
                            </div>

                            <div class="bg-white rounded-lg p-4">
                                <h5 class="font-bold text-pink-800 mb-3 flex items-center">
                                    <span class="text-2xl mr-2">🎓</span>
                                    学生出行建议
                                </h5>
                                <ul class="space-y-2 text-gray-700 text-sm">
                                    <li>✓ 优先选择地铁（5折优惠）</li>
                                    <li>✓ 短途可选共享单车</li>
                                    <li>✓ 办理学生卡享受更多优惠</li>
                                    <li>✓ 合理规划路线节省费用</li>
                                    <li class="text-blue-600">💰 每月可节省50%交通费</li>
                                </ul>
                            </div>
                        </div>

                        <div class="bg-yellow-50 border-l-4 border-yellow-400 rounded-lg p-4">
                            <h5 class="font-bold text-yellow-800 mb-2 flex items-center">
                                <span class="text-xl mr-2">🎓</span>
                                数学知识点
                            </h5>
                            <ul class="space-y-1 text-gray-700 text-sm">
                                <li>• <strong>分段函数：</strong>出租车和地铁采用分段计费，体现了分段函数的实际应用</li>
                                <li>• <strong>一次函数：</strong>共享单车费用 y = 0.6x 是典型的正比例函数</li>
                                <li>• <strong>函数比较：</strong>通过图像和数据对比，找出不同区间的最优方案</li>
                                <li>• <strong>实际应用：</strong>学会用数学方法分析和解决生活中的决策问题</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // 生成对比表格HTML
    generateComparisonTable() {
        const data = this.generateComparisonData(30);
        let html = '';

        data.forEach(row => {
            // 找出最便宜的方案
            let minCost = Infinity;
            let bestPlan = '';
            this.selectedPlans.forEach(plan => {
                const cost = parseFloat(row[plan]);
                if (cost < minCost) {
                    minCost = cost;
                    bestPlan = plan;
                }
            });

            html += `<tr>`;
            html += `<td class="text-center">${row.distance}</td>`;
            
            this.selectedPlans.forEach(plan => {
                const isBest = plan === bestPlan;
                const className = isBest ? 'bg-green-100 font-bold text-green-700' : '';
                html += `<td class="text-center ${className}">¥${row[plan]}</td>`;
            });

            html += `<td class="text-center font-bold text-indigo-700">${row.cheapest}</td>`;
            html += `</tr>`;
        });

        return html;
    }

    // 获取函数表达式
    getFormula(planType) {
        switch(planType) {
            case 'taxi':
                return 'y = 10 (x≤2) 或 y = 2.7x + 4.6 (x>2)';
            case 'bike':
                return 'y = 0.6x';
            case 'subway':
                return '分段计价（见规则说明）';
            case 'subwayStudent':
                return '分段计价 × 0.5（学生5折）';
            default:
                return '';
        }
    }
}
