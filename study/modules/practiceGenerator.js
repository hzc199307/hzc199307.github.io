// 练习生成模块
export class PracticeGenerator {
    constructor() {
        this.questionBank = this.initQuestionBank();
    }

    initQuestionBank() {
        return {
            basic: [
                {
                    knowledgePoint: '基础概念',
                    difficulty: 'easy',
                    type: 'choice',
                    question: '下列函数中，是一次函数的是（　）',
                    options: ['y = 2x + 1', 'y = x² + 1', 'y = 1/x', 'y = 2'],
                    answer: 0,
                    explanation: '一次函数的形式是 y = kx + b（k≠0），只有A选项符合'
                },
                {
                    knowledgePoint: '基础概念',
                    difficulty: 'easy',
                    type: 'choice',
                    question: '一次函数 y = 3x - 2 中，斜率k和截距b分别是（　）',
                    options: ['k=3, b=-2', 'k=-2, b=3', 'k=3, b=2', 'k=-3, b=-2'],
                    answer: 0,
                    explanation: '对比 y = kx + b，可知 k = 3，b = -2'
                },
                {
                    knowledgePoint: '函数求值',
                    difficulty: 'easy',
                    type: 'input',
                    question: '已知 y = 2x + 5，当 x = 3 时，y = ___',
                    answer: '11',
                    explanation: '将 x = 3 代入：y = 2×3 + 5 = 6 + 5 = 11'
                },
                {
                    knowledgePoint: '函数求值',
                    difficulty: 'easy',
                    type: 'input',
                    question: '已知 y = -x + 4，当 y = 0 时，x = ___',
                    answer: '4',
                    explanation: '令 y = 0：0 = -x + 4，解得 x = 4'
                },
                {
                    knowledgePoint: '表达式求解',
                    difficulty: 'medium',
                    type: 'input',
                    question: '一次函数经过点(0, 2)和点(1, 5)，则该函数表达式为 y = ___',
                    answer: '3x+2',
                    explanation: 'k = (5-2)/(1-0) = 3，过点(0,2)知b=2，所以 y = 3x + 2'
                }
            ],
            intermediate: [
                {
                    knowledgePoint: '图像性质',
                    difficulty: 'medium',
                    type: 'choice',
                    question: '一次函数 y = -2x + 3 的图像经过（　）象限',
                    options: ['一、二、三', '一、二、四', '一、三、四', '二、三、四'],
                    answer: 1,
                    explanation: 'k=-2<0, b=3>0，所以经过一、二、四象限'
                },
                {
                    knowledgePoint: '图像性质',
                    difficulty: 'medium',
                    type: 'choice',
                    question: '若一次函数 y = kx + 2 的图像经过第一、二、三象限，则k的取值范围是（　）',
                    options: ['k > 0', 'k < 0', 'k > 2', 'k < 2'],
                    answer: 0,
                    explanation: '经过一、二、三象限，说明k>0且b>0，已知b=2>0，所以k>0'
                },
                {
                    knowledgePoint: '增减性',
                    difficulty: 'medium',
                    type: 'choice',
                    question: '若一次函数 y = (m-1)x + 3 是减函数，则m的取值范围是（　）',
                    options: ['m > 1', 'm < 1', 'm > 0', 'm < 0'],
                    answer: 1,
                    explanation: '减函数说明k<0，即m-1<0，所以m<1'
                },
                {
                    knowledgePoint: '坐标交点',
                    difficulty: 'medium',
                    type: 'input',
                    question: '一次函数 y = 2x - 4 与x轴的交点坐标是___（格式：(x,y)）',
                    answer: '(2,0)',
                    explanation: '令y=0：0=2x-4，解得x=2，所以交点为(2,0)'
                },
                {
                    knowledgePoint: '实际应用',
                    difficulty: 'medium',
                    type: 'input',
                    question: '出租车起步价10元，每公里2元。行驶x公里的费用y（元）的函数表达式是 y = ___',
                    answer: '2x+10',
                    explanation: '费用 = 每公里价格×公里数 + 起步价 = 2x + 10'
                }
            ],
            advanced: [
                {
                    knowledgePoint: '综合应用',
                    difficulty: 'hard',
                    type: 'choice',
                    question: '甲、乙两种商品，甲商品每件a元，乙商品每件b元。若购买甲商品x件，乙商品y件，共花费100元，则y与x的函数关系式是（　）',
                    options: ['y = (100-ax)/b', 'y = (100-bx)/a', 'y = 100-ax-b', 'y = ax+b-100'],
                    answer: 0,
                    explanation: 'ax + by = 100，解出y：by = 100-ax，y = (100-ax)/b'
                },
                {
                    knowledgePoint: '方案比较',
                    difficulty: 'hard',
                    type: 'choice',
                    question: '两种手机套餐：A套餐月租30元，每分钟0.2元；B套餐月租50元，每分钟0.1元。当通话时长为多少分钟时，两套餐费用相同？（　）',
                    options: ['100分钟', '150分钟', '200分钟', '250分钟'],
                    answer: 2,
                    explanation: '令0.2x+30=0.1x+50，解得0.1x=20，x=200分钟'
                },
                {
                    knowledgePoint: '图像分析',
                    difficulty: 'hard',
                    type: 'input',
                    question: '直线y=kx+b与直线y=2x-3平行，且过点(1,4)，则k=___，b=___（用逗号分隔）',
                    answer: '2,2',
                    explanation: '平行则k相同，k=2。过点(1,4)：4=2×1+b，得b=2'
                },
                {
                    knowledgePoint: '综合计算',
                    difficulty: 'hard',
                    type: 'input',
                    question: '一次函数y=kx+b的图像经过点A(2,3)和B(-1,0)，则k+b的值是___',
                    answer: '2',
                    explanation: 'k=(3-0)/(2-(-1))=1，代入B点：0=-1+b，b=1，所以k+b=2'
                }
            ]
        };
    }

    generateByLevel(level) {
        const levelMap = {
            'basic': 'basic',
            'intermediate': 'intermediate',
            'advanced': 'advanced'
        };
        
        const questions = this.questionBank[levelMap[level]];
        return this.shuffleArray([...questions]).slice(0, 5);
    }

    generateCustom(knowledgePoints, count) {
        let allQuestions = [];
        
        // 根据知识点筛选题目
        Object.values(this.questionBank).forEach(levelQuestions => {
            levelQuestions.forEach(q => {
                if (this.matchKnowledgePoint(q.knowledgePoint, knowledgePoints)) {
                    allQuestions.push(q);
                }
            });
        });

        // 随机选择指定数量的题目
        const shuffled = this.shuffleArray(allQuestions);
        return shuffled.slice(0, Math.min(count, shuffled.length));
    }

    matchKnowledgePoint(questionPoint, selectedPoints) {
        const pointMap = {
            'expression': ['基础概念', '表达式求解', '函数求值'],
            'graph': ['图像性质', '坐标交点', '图像分析'],
            'properties': ['增减性', '图像性质'],
            'application': ['实际应用', '综合应用', '方案比较', '综合计算']
        };

        for (let point of selectedPoints) {
            if (pointMap[point] && pointMap[point].includes(questionPoint)) {
                return true;
            }
        }
        return false;
    }

    checkAnswer(question, userAnswer) {
        let correct = false;
        let feedback = '';

        if (question.type === 'choice') {
            correct = userAnswer === question.answer;
            if (correct) {
                feedback = `
                    <div class="correct-answer">
                        <h4 class="text-lg font-bold text-green-800 mb-2">✓ 回答正确！</h4>
                        <p class="text-gray-700"><strong>解析：</strong>${question.explanation}</p>
                    </div>
                `;
            } else {
                feedback = `
                    <div class="error-feedback">
                        <h4 class="text-lg font-bold text-red-800 mb-2">✗ 答案不正确</h4>
                        <p class="text-gray-700 mb-2">正确答案是：<strong>${String.fromCharCode(65 + question.answer)}</strong></p>
                        <p class="text-gray-700"><strong>解析：</strong>${question.explanation}</p>
                    </div>
                `;
            }
        } else {
            // 输入题，简单的字符串匹配
            const normalizedUser = userAnswer.replace(/\s/g, '').toLowerCase();
            const normalizedAnswer = question.answer.replace(/\s/g, '').toLowerCase();
            correct = normalizedUser === normalizedAnswer || normalizedUser.includes(normalizedAnswer);

            if (correct) {
                feedback = `
                    <div class="correct-answer">
                        <h4 class="text-lg font-bold text-green-800 mb-2">✓ 回答正确！</h4>
                        <p class="text-gray-700"><strong>解析：</strong>${question.explanation}</p>
                    </div>
                `;
            } else {
                feedback = `
                    <div class="error-feedback">
                        <h4 class="text-lg font-bold text-red-800 mb-2">✗ 答案不正确</h4>
                        <p class="text-gray-700 mb-2">正确答案是：<strong>${question.answer}</strong></p>
                        <p class="text-gray-700"><strong>解析：</strong>${question.explanation}</p>
                    </div>
                `;
            }
        }

        return { correct, feedback };
    }

    shuffleArray(array) {
        const newArray = [...array];
        for (let i = newArray.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
        }
        return newArray;
    }

    // 生成更多练习题的方法
    generateMoreQuestions(knowledgePoint, difficulty, count) {
        const templates = this.getQuestionTemplates(knowledgePoint, difficulty);
        const questions = [];

        for (let i = 0; i < count; i++) {
            const template = templates[i % templates.length];
            questions.push(this.fillTemplate(template));
        }

        return questions;
    }

    getQuestionTemplates(knowledgePoint, difficulty) {
        // 题目模板，可以通过随机参数生成不同的题目
        return [
            {
                type: 'template',
                generate: () => {
                    const k = Math.floor(Math.random() * 5) + 1;
                    const b = Math.floor(Math.random() * 10) - 5;
                    const x = Math.floor(Math.random() * 10);
                    const y = k * x + b;
                    return {
                        knowledgePoint: '函数求值',
                        difficulty: 'easy',
                        type: 'input',
                        question: `已知 y = ${k}x ${b >= 0 ? '+' : ''}${b}，当 x = ${x} 时，y = ___`,
                        answer: String(y),
                        explanation: `将 x = ${x} 代入：y = ${k}×${x} ${b >= 0 ? '+' : ''}${b} = ${k * x} ${b >= 0 ? '+' : ''}${b} = ${y}`
                    };
                }
            }
        ];
    }

    fillTemplate(template) {
        if (template.type === 'template') {
            return template.generate();
        }
        return template;
    }
}