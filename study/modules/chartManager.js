// 图表管理模块
export class ChartManager {
    constructor() {
        this.charts = {};
    }

    createKnowledgeChart(canvasId, knowledgeData) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // 销毁旧图表
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        const labels = Object.keys(knowledgeData);
        const data = Object.values(knowledgeData);

        this.charts[canvasId] = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: '掌握程度',
                    data: data,
                    backgroundColor: 'rgba(79, 70, 229, 0.2)',
                    borderColor: 'rgba(79, 70, 229, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(79, 70, 229, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(79, 70, 229, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20
                        },
                        pointLabels: {
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.r + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    createDailyChart(canvasId, dailyData) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // 销毁旧图表
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        const labels = this.getLast7Days();

        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '练习题数',
                    data: dailyData,
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(139, 92, 246, 0.8)',
                        'rgba(236, 72, 153, 0.8)',
                        'rgba(34, 197, 94, 0.8)',
                        'rgba(239, 68, 68, 0.8)'
                    ],
                    borderColor: [
                        'rgba(59, 130, 246, 1)',
                        'rgba(16, 185, 129, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(139, 92, 246, 1)',
                        'rgba(236, 72, 153, 1)',
                        'rgba(34, 197, 94, 1)',
                        'rgba(239, 68, 68, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return '完成 ' + context.parsed.y + ' 题';
                            }
                        }
                    }
                }
            }
        });
    }

    createProgressChart(canvasId, progressData) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // 销毁旧图表
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: progressData.dates,
                datasets: [{
                    label: '正确率',
                    data: progressData.accuracyRates,
                    borderColor: 'rgba(16, 185, 129, 1)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    createComparisonChart(canvasId, data1, data2, labels) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // 销毁旧图表
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: '你的成绩',
                        data: data1,
                        backgroundColor: 'rgba(79, 70, 229, 0.8)',
                        borderColor: 'rgba(79, 70, 229, 1)',
                        borderWidth: 2
                    },
                    {
                        label: '平均水平',
                        data: data2,
                        backgroundColor: 'rgba(156, 163, 175, 0.8)',
                        borderColor: 'rgba(156, 163, 175, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    getLast7Days() {
        const days = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
        const result = [];
        const today = new Date();

        for (let i = 6; i >= 0; i--) {
            const date = new Date(today);
            date.setDate(date.getDate() - i);
            const dayName = days[date.getDay()];
            const dateStr = `${date.getMonth() + 1}/${date.getDate()}`;
            result.push(i === 0 ? '今天' : `${dayName}`);
        }

        return result;
    }

    destroyChart(canvasId) {
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
            delete this.charts[canvasId];
        }
    }

    destroyAllCharts() {
        Object.keys(this.charts).forEach(canvasId => {
            this.destroyChart(canvasId);
        });
    }

    updateChart(canvasId, newData) {
        if (this.charts[canvasId]) {
            const chart = this.charts[canvasId];
            
            if (chart.data.datasets && chart.data.datasets[0]) {
                chart.data.datasets[0].data = newData;
                chart.update();
            }
        }
    }

    // 创建函数图像
    createFunctionGraph(canvasId, k, b, xMin = -10, xMax = 10) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // 销毁旧图表
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // 生成数据点
        const points = [];
        const step = (xMax - xMin) / 100;
        for (let x = xMin; x <= xMax; x += step) {
            points.push({ x: x, y: k * x + b });
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: `y = ${k}x ${b >= 0 ? '+' : ''}${b}`,
                    data: points,
                    borderColor: 'rgba(59, 130, 246, 1)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'center',
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'center',
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
}
