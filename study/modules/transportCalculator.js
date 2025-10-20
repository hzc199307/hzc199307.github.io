// 交通出行方案计算器模块
export class TransportCalculator {
    constructor() {
        this.planA = { base: 10, rate: 2.4, name: '方案A' };
        this.planB = { base: 2, rate: 0.5, name: '方案B' };
        this.chart = null;
    }

    // 更新方案参数
    updatePlan(plan, base, rate) {
        if (plan === 'A') {
            this.planA.base = parseFloat(base);
            this.planA.rate = parseFloat(rate);
        } else if (plan === 'B') {
            this.planB.base = parseFloat(base);
            this.planB.rate = parseFloat(rate);
        }
    }

    // 计算指定距离的费用
    calculateCost(plan, distance) {
        if (plan === 'A') {
            return this.planA.rate * distance + this.planA.base;
        } else if (plan === 'B') {
            return this.planB.rate * distance + this.planB.base;
        }
        return 0;
    }

    // 计算两个方案的交点（费用相等时的距离）
    calculateIntersection() {
        // y1 = k1*x + b1
        // y2 = k2*x + b2
        // k1*x + b1 = k2*x + b2
        // (k1 - k2)*x = b2 - b1
        // x = (b2 - b1) / (k1 - k2)

        const k1 = this.planA.rate;
        const b1 = this.planA.base;
        const k2 = this.planB.rate;
        const b2 = this.planB.base;

        // 如果斜率相同，没有交点
        if (Math.abs(k1 - k2) < 0.0001) {
            return null;
        }

        const x = (b2 - b1) / (k1 - k2);
        const y = k1 * x + b1;

        return { distance: x, cost: y };
    }

    // 生成对比数据
    generateComparisonData(maxDistance = 20) {
        const data = [];
        const step = maxDistance / 10;

        for (let distance = 0; distance <= maxDistance; distance += step) {
            const costA = this.calculateCost('A', distance);
            const costB = this.calculateCost('B', distance);
            const diff = Math.abs(costA - costB);
            const cheaper = costA < costB ? '方案A' : (costB < costA ? '方案B' : '相同');

            data.push({
                distance: distance.toFixed(1),
                costA: costA.toFixed(2),
                costB: costB.toFixed(2),
                diff: diff.toFixed(2),
                cheaper: cheaper
            });
        }

        return data;
    }

    // 生成图表数据
    generateChartData(maxDistance = 20) {
        const labels = [];
        const dataA = [];
        const dataB = [];
        const step = 0.5;

        for (let distance = 0; distance <= maxDistance; distance += step) {
            labels.push(distance.toFixed(1));
            dataA.push(this.calculateCost('A', distance).toFixed(2));
            dataB.push(this.calculateCost('B', distance).toFixed(2));
        }

        return { labels, dataA, dataB };
    }

    // 创建对比图表
    createComparisonChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // 销毁旧图表
        if (this.chart) {
            this.chart.destroy();
        }

        const chartData = this.generateChartData(20);
        const intersection = this.calculateIntersection();

        // 准备交点标注
        const annotations = {};
        if (intersection && intersection.distance > 0 && intersection.distance <= 20) {
            annotations.intersection = {
                type: 'point',
                xValue: intersection.distance.toFixed(1),
                yValue: intersection.cost.toFixed(2),
                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                borderColor: 'rgb(255, 99, 132)',
                borderWidth: 2,
                radius: 8,
                label: {
                    enabled: true,
                    content: `交点: ${intersection.distance.toFixed(1)}km`,
                    position: 'top'
                }
            };
        }

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [
                    {
                        label: '方案A费用',
                        data: chartData.dataA,
                        borderColor: 'rgba(34, 197, 94, 1)',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        borderWidth: 3,
                        tension: 0,
                        pointRadius: 0,
                        pointHoverRadius: 6
                    },
                    {
                        label: '方案B费用',
                        data: chartData.dataB,
                        borderColor: 'rgba(59, 130, 246, 1)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 3,
                        tension: 0,
                        pointRadius: 0,
                        pointHoverRadius: 6
                    }
                ]
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

    // 生成交点分析HTML
    generateIntersectionAnalysis() {
        const intersection = this.calculateIntersection();

        if (!intersection || intersection.distance <= 0) {
            return `
                <div class="flex items-start">
                    <div class="text-4xl mr-4">⚠️</div>
                    <div>
                        <h4 class="text-xl font-bold text-yellow-800 mb-2">无交点情况</h4>
                        <p class="text-gray-700 mb-2">两个方案的费用曲线没有交点，说明：</p>
                        <ul class="list-disc list-inside text-gray-700 space-y-1">
                            <li>两条直线平行（斜率相同）或</li>
                            <li>在考察范围内始终是一个方案更优</li>
                        </ul>
                        <div class="mt-4 p-4 bg-white rounded-lg">
                            <p class="font-bold text-gray-800">
                                ${this.planA.rate < this.planB.rate ? '方案A的每公里价格更低' : '方案B的每公里价格更低'}，
                                ${this.planA.base < this.planB.base ? '且起步价也更低' : '但起步价较高'}
                            </p>
                        </div>
                    </div>
                </div>
            `;
        }

        const distance = intersection.distance.toFixed(2);
        const cost = intersection.cost.toFixed(2);

        return `
            <div class="flex items-start">
                <div class="text-4xl mr-4">🎯</div>
                <div class="flex-1">
                    <h4 class="text-xl font-bold text-yellow-800 mb-3">交点分析（关键决策点）</h4>
                    
                    <div class="bg-white rounded-lg p-4 mb-4">
                        <p class="text-lg text-gray-800 mb-2">
                            <strong>交点位置：</strong>距离 = <span class="text-2xl font-bold text-indigo-600">${distance}</span> 公里
                        </p>
                        <p class="text-lg text-gray-800">
                            <strong>此时费用：</strong><span class="text-2xl font-bold text-indigo-600">¥${cost}</span> 元
                        </p>
                    </div>

                    <div class="bg-indigo-50 rounded-lg p-4 mb-4">
                        <h5 class="font-bold text-indigo-800 mb-2">📐 数学原理：</h5>
                        <p class="text-gray-700 mb-2">令两个方案费用相等：</p>
                        <div class="bg-white rounded p-3 font-mono text-sm mb-2">
                            ${this.planA.rate}x + ${this.planA.base} = ${this.planB.rate}x + ${this.planB.base}
                        </div>
                        <p class="text-gray-700 mb-2">解方程：</p>
                        <div class="bg-white rounded p-3 font-mono text-sm mb-2">
                            ${(this.planA.rate - this.planB.rate).toFixed(2)}x = ${(this.planB.base - this.planA.base).toFixed(2)}
                        </div>
                        <div class="bg-white rounded p-3 font-mono text-sm">
                            x = ${distance} 公里
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-green-50 border-2 border-green-400 rounded-lg p-4">
                            <h5 class="font-bold text-green-800 mb-2">✓ 距离 &lt; ${distance} 公里时</h5>
                            <p class="text-gray-700">
                                ${this.planA.base < this.planB.base ? '方案A' : '方案B'}更省钱<br>
                                <span class="text-sm">(起步价低的方案占优势)</span>
                            </p>
                        </div>
                        <div class="bg-blue-50 border-2 border-blue-400 rounded-lg p-4">
                            <h5 class="font-bold text-blue-800 mb-2">✓ 距离 &gt; ${distance} 公里时</h5>
                            <p class="text-gray-700">
                                ${this.planA.rate < this.planB.rate ? '方案A' : '方案B'}更省钱<br>
                                <span class="text-sm">(每公里价格低的方案占优势)</span>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // 生成对比表格HTML
    generateComparisonTable() {
        const data = this.generateComparisonData(20);
        let html = '';

        data.forEach(row => {
            const highlightA = row.cheaper === '方案A' ? 'bg-green-100 font-bold' : '';
            const highlightB = row.cheaper === '方案B' ? 'bg-blue-100 font-bold' : '';
            const highlightSame = row.cheaper === '相同' ? 'bg-yellow-100 font-bold' : '';

            html += `
                <tr class="${highlightSame}">
                    <td class="text-center">${row.distance}</td>
                    <td class="text-center ${highlightA}">¥${row.costA}</td>
                    <td class="text-center ${highlightB}">¥${row.costB}</td>
                    <td class="text-center">¥${row.diff}</td>
                    <td class="text-center font-bold ${row.cheaper === '方案A' ? 'text-green-700' : (row.cheaper === '方案B' ? 'text-blue-700' : 'text-yellow-700')}">${row.cheaper}</td>
                </tr>
            `;
        });

        return html;
    }

    // 生成决策建议HTML
    generateDecisionAdvice() {
        const intersection = this.calculateIntersection();
        const avgDistance = 10; // 假设平均出行距离

        let advice = '';
        let recommendation = '';

        if (!intersection || intersection.distance <= 0) {
            // 没有交点的情况
            const costA10 = this.calculateCost('A', avgDistance);
            const costB10 = this.calculateCost('B', avgDistance);
            
            if (costA10 < costB10) {
                recommendation = '方案A';
                advice = `在所有距离下，方案A都更经济实惠。建议选择方案A。`;
            } else {
                recommendation = '方案B';
                advice = `在所有距离下，方案B都更经济实惠。建议选择方案B。`;
            }
        } else {
            // 有交点的情况
            const breakpoint = intersection.distance;
            
            if (avgDistance < breakpoint) {
                recommendation = this.planA.base < this.planB.base ? '方案A' : '方案B';
                advice = `对于${avgDistance}公里的出行，${recommendation}更省钱。如果你的日常出行距离较短（小于${breakpoint.toFixed(1)}公里），建议选择${recommendation}。`;
            } else {
                recommendation = this.planA.rate < this.planB.rate ? '方案A' : '方案B';
                advice = `对于${avgDistance}公里的出行，${recommendation}更省钱。如果你的日常出行距离较长（大于${breakpoint.toFixed(1)}公里），建议选择${recommendation}。`;
            }
        }

        return `
            <div class="flex items-start">
                <div class="text-5xl mr-4">💡</div>
                <div class="flex-1">
                    <h4 class="text-2xl font-bold text-green-800 mb-4">智能决策建议</h4>
                    
                    <div class="bg-white rounded-lg p-6 mb-4">
                        <div class="text-center mb-4">
                            <div class="text-4xl font-bold text-indigo-600 mb-2">${recommendation}</div>
                            <div class="text-xl text-gray-700">推荐方案</div>
                        </div>
                        <p class="text-lg text-gray-700 text-center">${advice}</p>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-white rounded-lg p-4">
                            <h5 class="font-bold text-gray-800 mb-3">📊 方案A特点：</h5>
                            <ul class="space-y-2 text-gray-700">
                                <li>• 起步价：¥${this.planA.base}</li>
                                <li>• 每公里：¥${this.planA.rate}</li>
                                <li>• 函数：y = ${this.planA.rate}x + ${this.planA.base}</li>
                                <li class="text-sm text-gray-600">
                                    ${this.planA.base < this.planB.base ? '✓ 起步价较低，适合短途' : '✗ 起步价较高'}
                                </li>
                                <li class="text-sm text-gray-600">
                                    ${this.planA.rate < this.planB.rate ? '✓ 单价较低，适合长途' : '✗ 单价较高'}
                                </li>
                            </ul>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <h5 class="font-bold text-gray-800 mb-3">📊 方案B特点：</h5>
                            <ul class="space-y-2 text-gray-700">
                                <li>• 起步价：¥${this.planB.base}</li>
                                <li>• 每公里：¥${this.planB.rate}</li>
                                <li>• 函数：y = ${this.planB.rate}x + ${this.planB.base}</li>
                                <li class="text-sm text-gray-600">
                                    ${this.planB.base < this.planA.base ? '✓ 起步价较低，适合短途' : '✗ 起步价较高'}
                                </li>
                                <li class="text-sm text-gray-600">
                                    ${this.planB.rate < this.planA.rate ? '✓ 单价较低，适合长途' : '✗ 单价较高'}
                                </li>
                            </ul>
                        </div>
                    </div>

                    <div class="bg-yellow-50 border-l-4 border-yellow-400 rounded-lg p-4 mt-4">
                        <h5 class="font-bold text-yellow-800 mb-2">🎓 学习要点：</h5>
                        <ul class="space-y-1 text-gray-700 text-sm">
                            <li>• 一次函数 y = kx + b 中，b（截距）代表起步价，k（斜率）代表每单位的变化率</li>
                            <li>• 两条直线的交点表示两种方案费用相等的临界点</li>
                            <li>• 斜率越大，直线越陡，表示费用增长越快</li>
                            <li>• 通过比较函数图像，可以直观地做出最优决策</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    // 获取函数表达式
    getFormula(plan) {
        if (plan === 'A') {
            return `y = ${this.planA.rate}x + ${this.planA.base}`;
        } else if (plan === 'B') {
            return `y = ${this.planB.rate}x + ${this.planB.base}`;
        }
        return '';
    }
}
