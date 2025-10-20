// 解题辅导模块
export class ProblemSolver {
    constructor() {
        this.problems = {
            basic: this.getBasicProblems(),
            graph: this.getGraphProblems(),
            application: this.getApplicationProblems(),
            comprehensive: this.getComprehensiveProblems()
        };
    }

    getProblem(type) {
        const problemList = this.problems[type];
        return problemList[Math.floor(Math.random() * problemList.length)];
    }

    getBasicProblems() {
        return [
            {
                id: 'basic1',
                type: 'basic',
                question: `
                    <p class="mb-3">已知一次函数 y = 2x + 3</p>
                    <p>(1) 求当 x = 5 时，y 的值</p>
                    <p>(2) 求当 y = 11 时，x 的值</p>
                `,
                answer: 'x=5时y=13; y=11时x=4',
                hint: '把x或y的值代入函数表达式，然后计算另一个变量的值',
                solution: `
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>求 x = 5 时的 y 值</strong><br>
                            将 x = 5 代入 y = 2x + 3<br>
                            y = 2 × 5 + 3 = 10 + 3 = 13
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>求 y = 11 时的 x 值</strong><br>
                            将 y = 11 代入 y = 2x + 3<br>
                            11 = 2x + 3<br>
                            2x = 11 - 3 = 8<br>
                            x = 4
                        </div>
                    </div>
                    <div class="correct-answer">
                        <strong>✓ 答案：</strong>(1) y = 13；(2) x = 4
                    </div>
                `
            },
            {
                id: 'basic2',
                type: 'basic',
                question: `
                    <p class="mb-3">已知一次函数经过点 (0, 3) 和点 (2, 7)</p>
                    <p>求这个一次函数的表达式</p>
                `,
                answer: 'y=2x+3',
                hint: '先用两点求斜率k，再用其中一点求截距b',
                solution: `
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>设函数表达式</strong><br>
                            设一次函数为 y = kx + b
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>求斜率k</strong><br>
                            k = (y₂ - y₁) / (x₂ - x₁)<br>
                            k = (7 - 3) / (2 - 0) = 4 / 2 = 2
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">3</span>
                        <div class="solution-step-content">
                            <strong>求截距b</strong><br>
                            因为过点 (0, 3)，所以 b = 3<br>
                            （或将 (2, 7) 代入 y = 2x + b，得 7 = 4 + b，b = 3）
                        </div>
                    </div>
                    <div class="correct-answer">
                        <strong>✓ 答案：</strong>y = 2x + 3
                    </div>
                `
            },
            {
                id: 'basic3',
                type: 'basic',
                question: `
                    <p class="mb-3">若一次函数 y = (m-2)x + 3 是增函数</p>
                    <p>求 m 的取值范围</p>
                `,
                answer: 'm>2',
                hint: '一次函数是增函数，说明斜率k>0',
                solution: `
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>理解题意</strong><br>
                            一次函数是增函数 ⟺ 斜率 k > 0
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>列不等式</strong><br>
                            因为 k = m - 2<br>
                            所以 m - 2 > 0
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">3</span>
                        <div class="solution-step-content">
                            <strong>求解</strong><br>
                            m > 2
                        </div>
                    </div>
                    <div class="correct-answer">
                        <strong>✓ 答案：</strong>m > 2
                    </div>
                `
            }
        ];
    }

    getGraphProblems() {
        return [
            {
                id: 'graph1',
                type: 'graph',
                question: `
                    <p class="mb-3">一次函数 y = -2x + 4 的图像</p>
                    <p>(1) 经过哪几个象限？</p>
                    <p>(2) 与x轴、y轴的交点坐标分别是多少？</p>
                `,
                answer: '一二四象限; x轴(2,0), y轴(0,4)',
                hint: 'k<0, b>0，可以判断象限；令x=0求y轴交点，令y=0求x轴交点',
                solution: `
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>判断经过的象限</strong><br>
                            因为 k = -2 < 0，b = 4 > 0<br>
                            所以图像经过<strong>一、二、四象限</strong>
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>求与y轴交点</strong><br>
                            令 x = 0，得 y = 4<br>
                            与y轴交点为 <strong>(0, 4)</strong>
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">3</span>
                        <div class="solution-step-content">
                            <strong>求与x轴交点</strong><br>
                            令 y = 0，得 0 = -2x + 4<br>
                            2x = 4，x = 2<br>
                            与x轴交点为 <strong>(2, 0)</strong>
                        </div>
                    </div>
                    <div class="correct-answer">
                        <strong>✓ 答案：</strong>(1) 一、二、四象限；(2) 与x轴交点(2, 0)，与y轴交点(0, 4)
                    </div>
                `
            },
            {
                id: 'graph2',
                type: 'graph',
                question: `
                    <p class="mb-3">直线 y = kx + b 经过点 A(1, 3) 和点 B(-2, 0)</p>
                    <p>(1) 求这条直线的表达式</p>
                    <p>(2) 判断点 C(2, 4) 是否在这条直线上</p>
                `,
                answer: 'y=x+2; 点C在直线上',
                hint: '用两点求出k和b，然后把点C坐标代入验证',
                solution: `
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>求斜率k</strong><br>
                            k = (3 - 0) / (1 - (-2)) = 3 / 3 = 1
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>求截距b</strong><br>
                            将点 A(1, 3) 代入 y = x + b<br>
                            3 = 1 + b，得 b = 2<br>
                            所以直线表达式为 <strong>y = x + 2</strong>
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">3</span>
                        <div class="solution-step-content">
                            <strong>验证点C</strong><br>
                            将 x = 2 代入 y = x + 2<br>
                            得 y = 2 + 2 = 4<br>
                            因为计算结果与点C的纵坐标相同<br>
                            所以<strong>点C在这条直线上</strong>
                        </div>
                    </div>
                    <div class="correct-answer">
                        <strong>✓ 答案：</strong>(1) y = x + 2；(2) 点C在直线上
                    </div>
                `
            }
        ];
    }

    getApplicationProblems() {
        return [
            {
                id: 'app1',
                type: 'application',
                question: `
                    <p class="mb-3"><strong>出租车计费问题</strong></p>
                    <p>某市出租车收费标准：起步价8元（3公里以内），超过3公里后每公里加收2.4元。</p>
                    <p>(1) 写出行驶x公里（x>3）时的车费y（元）与x的函数关系式</p>
                    <p>(2) 小明乘出租车行驶了10公里，应付多少车费？</p>
                `,
                answer: 'y=2.4x+0.8; 24.8元',
                hint: '起步价包含了前3公里，超过部分才按每公里2.4元计算',
                solution: `
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>分析题意</strong><br>
                            • 前3公里：8元（固定）<br>
                            • 超过3公里的部分：每公里2.4元<br>
                            • 超过的公里数：(x - 3) 公里
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>建立函数关系式</strong><br>
                            y = 8 + 2.4(x - 3)<br>
                            y = 8 + 2.4x - 7.2<br>
                            <strong>y = 2.4x + 0.8</strong> (x > 3)
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">3</span>
                        <div class="solution-step-content">
                            <strong>计算10公里的车费</strong><br>
                            将 x = 10 代入 y = 2.4x + 0.8<br>
                            y = 2.4 × 10 + 0.8 = 24 + 0.8 = <strong>24.8元</strong>
                        </div>
                    </div>
                    <div class="correct-answer">
                        <strong>✓ 答案：</strong>(1) y = 2.4x + 0.8 (x > 3)；(2) 24.8元
                    </div>
                `
            },
            {
                id: 'app2',
                type: 'application',
                question: `
                    <p class="mb-3"><strong>手机套餐选择问题</strong></p>
                    <p>移动公司推出两种套餐：</p>
                    <p>套餐A：月租30元，每分钟通话费0.2元</p>
                    <p>套餐B：月租50元，每分钟通话费0.1元</p>
                    <p>(1) 分别写出两种套餐的费用y（元）与通话时长x（分钟）的函数关系式</p>
                    <p>(2) 通话多少分钟时，两种套餐费用相同？</p>
                    <p>(3) 如果每月通话300分钟，选择哪种套餐更省钱？</p>
                `,
                answer: 'A: y=0.2x+30, B: y=0.1x+50; 200分钟; 选B',
                hint: '建立两个函数，令它们相等求交点，然后比较300分钟时的费用',
                solution: `
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>建立函数关系式</strong><br>
                            套餐A：y = 0.2x + 30<br>
                            套餐B：y = 0.1x + 50
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>求费用相同时的通话时长</strong><br>
                            令 0.2x + 30 = 0.1x + 50<br>
                            0.2x - 0.1x = 50 - 30<br>
                            0.1x = 20<br>
                            x = <strong>200分钟</strong>
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">3</span>
                        <div class="solution-step-content">
                            <strong>比较300分钟时的费用</strong><br>
                            套餐A：y = 0.2 × 300 + 30 = 60 + 30 = 90元<br>
                            套餐B：y = 0.1 × 300 + 50 = 30 + 50 = 80元<br>
                            因为 80 < 90，所以<strong>选择套餐B更省钱</strong>
                        </div>
                    </div>
                    <div class="correct-answer">
                        <strong>✓ 答案：</strong>(1) A: y=0.2x+30, B: y=0.1x+50；(2) 200分钟；(3) 选择套餐B
                    </div>
                    <div class="hint-box mt-4">
                        <strong>💡 决策规律：</strong><br>
                        • 通话 < 200分钟：选A更省<br>
                        • 通话 = 200分钟：两者相同<br>
                        • 通话 > 200分钟：选B更省
                    </div>
                `
            }
        ];
    }

    getComprehensiveProblems() {
        return [
            {
                id: 'comp1',
                type: 'comprehensive',
                question: `
                    <p class="mb-3"><strong>综合应用题</strong></p>
                    <p>甲、乙两地相距300千米，一辆汽车从甲地出发前往乙地，出发2小时后距乙地还有180千米。</p>
                    <p>(1) 求汽车行驶的路程y（千米）与时间x（小时）的函数关系式</p>
                    <p>(2) 汽车出发多长时间后到达乙地？</p>
                    <p>(3) 出发3.5小时后，汽车距甲地多远？</p>
                `,
                answer: 'y=60x; 5小时; 210千米',
                hint: '先求速度，速度=路程/时间。2小时行驶了300-180=120千米',
                solution: `
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>求汽车速度</strong><br>
                            2小时后距乙地180千米<br>
                            说明2小时行驶了：300 - 180 = 120千米<br>
                            速度 = 120 ÷ 2 = 60千米/小时
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>建立函数关系式</strong><br>
                            因为汽车匀速行驶，从甲地出发<br>
                            所以 <strong>y = 60x</strong>
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">3</span>
                        <div class="solution-step-content">
                            <strong>求到达乙地的时间</strong><br>
                            令 y = 300<br>
                            60x = 300<br>
                            x = 5<br>
                            所以<strong>5小时</strong>后到达乙地
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">4</span>
                        <div class="solution-step-content">
                            <strong>求3.5小时后距甲地的距离</strong><br>
                            将 x = 3.5 代入 y = 60x<br>
                            y = 60 × 3.5 = <strong>210千米</strong>
                        </div>
                    </div>
                    <div class="correct-answer">
                        <strong>✓ 答案：</strong>(1) y = 60x；(2) 5小时；(3) 210千米
                    </div>
                `
            }
        ];
    }

    analyzeCustomProblem(problemText) {
        // 简单分析用户输入的题目
        return {
            id: 'custom',
            type: 'custom',
            question: `<p>${problemText}</p>`,
            answer: '需要具体分析',
            hint: '仔细读题，找出已知条件和未知量，思考它们之间的关系',
            solution: this.generateCustomSolution(problemText)
        };
    }

    generateCustomSolution(problemText) {
        return `
            <div class="bg-blue-50 rounded-xl p-6">
                <h4 class="text-xl font-bold text-blue-800 mb-4">解题思路指导</h4>
                <div class="space-y-4">
                    <div class="solution-step">
                        <span class="solution-step-number">1</span>
                        <div class="solution-step-content">
                            <strong>审题</strong><br>
                            仔细阅读题目，找出：<br>
                            • 已知条件是什么？<br>
                            • 要求什么？<br>
                            • 涉及哪些变量？
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">2</span>
                        <div class="solution-step-content">
                            <strong>建立模型</strong><br>
                            • 确定自变量x和因变量y<br>
                            • 找出k（变化率）和b（初始值）<br>
                            • 写出函数表达式 y = kx + b
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">3</span>
                        <div class="solution-step-content">
                            <strong>求解</strong><br>
                            • 根据题目要求代入计算<br>
                            • 或者画出图像辅助分析<br>
                            • 注意单位和实际意义
                        </div>
                    </div>
                    <div class="solution-step">
                        <span class="solution-step-number">4</span>
                        <div class="solution-step-content">
                            <strong>检验</strong><br>
                            • 答案是否符合实际情况？<br>
                            • 单位是否正确？<br>
                            • 可以用其他方法验证吗？
                        </div>
                    </div>
                </div>
                <div class="hint-box mt-6">
                    <strong>💡 提示：</strong>如果遇到困难，可以：<br>
                    • 画图帮助理解<br>
                    • 列表整理数据<br>
                    • 从简单情况开始分析<br>
                    • 寻找题目中的关键词（如"起步价"、"每...多少"等）
                </div>
            </div>
        `;
    }

    checkAnswer(problem, studentAnswer) {
        // 简化的答案检查
        const correctKeywords = problem.answer.toLowerCase().split(/[;,，；]/);
        const studentLower = studentAnswer.toLowerCase();
        
        let matchCount = 0;
        correctKeywords.forEach(keyword => {
            if (studentLower.includes(keyword.trim())) {
                matchCount++;
            }
        });

        const isCorrect = matchCount >= correctKeywords.length * 0.6;

        if (isCorrect) {
            return `
                <div class="correct-answer">
                    <h4 class="text-xl font-bold text-green-800 mb-3">🎉 太棒了！答对了！</h4>
                    <p class="text-gray-700 mb-4">你的思路很清晰，继续保持！</p>
                    ${problem.solution}
                </div>
            `;
        } else {
            return `
                <div class="error-feedback">
                    <h4 class="text-xl font-bold text-red-800 mb-3">💪 再想想看</h4>
                    <p class="text-gray-700 mb-4">
                        你的答案：${studentAnswer}<br>
                        还不太准确，但不要气馁！让我们一起分析：
                    </p>
                </div>
                <div class="hint-box">
                    <h4 class="text-lg font-bold text-yellow-800 mb-2">💡 提示</h4>
                    <p class="text-gray-700">${problem.hint}</p>
                </div>
                <div class="mt-4">
                    <button onclick="showSolution()" class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                        查看详细解答
                    </button>
                </div>
            `;
        }
    }

    getHint(problem) {
        return problem.hint;
    }

    getDetailedSolution(problem) {
        return problem.solution;
    }
}