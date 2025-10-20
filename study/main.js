// 导入模块
import { ConceptExplainer } from './modules/conceptExplainer.js';
import { ProblemSolver } from './modules/problemSolver.js';
import { PracticeGenerator } from './modules/practiceGenerator.js';
import { ProgressTracker } from './modules/progressTracker.js';
import { ChartManager } from './modules/chartManager.js';
import { TransportCalculator } from './modules/transportCalculator.js';

// 全局状态管理
const appState = {
    currentSection: 'home',
    currentConcept: 'basic',
    currentProblem: null,
    practiceQuestions: [],
    currentQuestionIndex: 0,
    userProgress: {
        conceptsLearned: 6,
        problemsSolved: 12,
        studyTime: 45,
        accuracyRate: 85,
        knowledgePoints: {
            '基础概念': 90,
            '斜率理解': 85,
            '截距理解': 80,
            '图像绘制': 75,
            '性质分析': 70,
            '实际应用': 65
        },
        dailyPractice: [3, 5, 4, 6, 8, 7, 5]
    }
};

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initConceptSection();
    initSolveSection();
    initPracticeSection();
    initProgressSection();
    showEncouragement('欢迎来到一次函数学习助手！让我们一起开始学习之旅吧！🎉');
});

// 导航功能
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const section = btn.dataset.section;
            navigateTo(section);
        });
    });
}

window.navigateTo = function(section) {
    // 更新导航按钮状态
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.section === section) {
            btn.classList.add('active');
        }
    });

    // 更新内容区域
    document.querySelectorAll('.section-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(section).classList.add('active');

    appState.currentSection = section;

    // 如果切换到进度页面，更新图表
    if (section === 'progress') {
        updateProgressCharts();
    }
};

// 概念讲解模块初始化
function initConceptSection() {
    const conceptButtons = document.querySelectorAll('.concept-btn');
    conceptButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            conceptButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const concept = btn.dataset.concept;
            loadConceptContent(concept);
        });
    });

    // 加载默认概念
    loadConceptContent('basic');
}

function loadConceptContent(concept) {
    const contentDiv = document.getElementById('concept-content');
    const explainer = new ConceptExplainer();
    const content = explainer.getExplanation(concept);
    contentDiv.innerHTML = content;
    
    // 更新进度
    appState.userProgress.conceptsLearned++;
    updateProgressDisplay();
}

window.generateCustomExplanation = function() {
    const question = document.getElementById('custom-question').value.trim();
    if (!question) {
        showEncouragement('请输入你想了解的问题哦！😊');
        return;
    }

    const contentDiv = document.getElementById('concept-content');
    const explainer = new ConceptExplainer();
    const content = explainer.generateCustomExplanation(question);
    contentDiv.innerHTML = content;
    
    showEncouragement('很好的问题！继续保持好奇心！🌟');
};

// 解题辅导模块初始化
function initSolveSection() {
    const problemTypeButtons = document.querySelectorAll('.problem-type-btn');
    problemTypeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            problemTypeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const type = btn.dataset.type;
            loadProblem(type);
        });
    });
}

function loadProblem(type) {
    const solver = new ProblemSolver();
    const problem = solver.getProblem(type);
    appState.currentProblem = problem;

    document.getElementById('problem-display').classList.remove('hidden');
    document.getElementById('problem-text').innerHTML = problem.question;
    document.getElementById('student-answer').value = '';
    document.getElementById('solution-display').classList.add('hidden');

    showEncouragement('加油！相信你一定能解决这道题！💪');
}

window.solveCustomProblem = function() {
    const problemText = document.getElementById('custom-problem').value.trim();
    if (!problemText) {
        showEncouragement('请输入题目内容哦！📝');
        return;
    }

    const solver = new ProblemSolver();
    const problem = solver.analyzeCustomProblem(problemText);
    appState.currentProblem = problem;

    document.getElementById('problem-display').classList.remove('hidden');
    document.getElementById('problem-text').innerHTML = problem.question;
    document.getElementById('student-answer').value = '';
    document.getElementById('solution-display').classList.add('hidden');
};

window.checkAnswer = function() {
    const studentAnswer = document.getElementById('student-answer').value.trim();
    if (!studentAnswer) {
        showEncouragement('写下你的想法，不要害怕犯错！✍️');
        return;
    }

    const solver = new ProblemSolver();
    const feedback = solver.checkAnswer(appState.currentProblem, studentAnswer);
    
    const solutionDiv = document.getElementById('solution-display');
    const contentDiv = document.getElementById('solution-content');
    
    contentDiv.innerHTML = feedback;
    solutionDiv.classList.remove('hidden');

    // 更新进度
    appState.userProgress.problemsSolved++;
    updateProgressDisplay();

    if (feedback.includes('correct-answer')) {
        showEncouragement('太棒了！你答对了！继续保持！🎉');
    } else {
        showEncouragement('没关系，错误是学习的一部分！再试试看！💪');
    }
};

window.showHint = function() {
    if (!appState.currentProblem) return;

    const solver = new ProblemSolver();
    const hint = solver.getHint(appState.currentProblem);
    
    const solutionDiv = document.getElementById('solution-display');
    const contentDiv = document.getElementById('solution-content');
    
    contentDiv.innerHTML = `<div class="hint-box">
        <h4 class="text-lg font-bold text-yellow-800 mb-2">💡 提示</h4>
        <p class="text-gray-700">${hint}</p>
    </div>`;
    solutionDiv.classList.remove('hidden');

    showEncouragement('这个提示应该能帮到你！加油！🌟');
};

window.showSolution = function() {
    if (!appState.currentProblem) return;

    const solver = new ProblemSolver();
    const solution = solver.getDetailedSolution(appState.currentProblem);
    
    const solutionDiv = document.getElementById('solution-display');
    const contentDiv = document.getElementById('solution-content');
    
    contentDiv.innerHTML = solution;
    solutionDiv.classList.remove('hidden');

    showEncouragement('仔细看看解题步骤，理解思路最重要！📖');
};

// 练习生成模块初始化
function initPracticeSection() {
    // 初始化已在HTML中通过onclick完成
}

window.generatePractice = function(level) {
    const generator = new PracticeGenerator();
    const questions = generator.generateByLevel(level);
    appState.practiceQuestions = questions;
    appState.currentQuestionIndex = 0;

    displayPracticeQuestion();
    showEncouragement('练习题已生成！加油做题吧！📝');
};

window.generateCustomPractice = function() {
    const selectedPoints = Array.from(document.querySelectorAll('.knowledge-point:checked'))
        .map(cb => cb.value);
    const count = parseInt(document.getElementById('practice-count').value);

    if (selectedPoints.length === 0) {
        showEncouragement('请至少选择一个知识点哦！😊');
        return;
    }

    const generator = new PracticeGenerator();
    const questions = generator.generateCustom(selectedPoints, count);
    appState.practiceQuestions = questions;
    appState.currentQuestionIndex = 0;

    displayPracticeQuestion();
    showEncouragement('定制练习已生成！加油！💪');
};

function displayPracticeQuestion() {
    const displayDiv = document.getElementById('practice-display');
    const contentDiv = document.getElementById('practice-content');
    
    displayDiv.classList.remove('hidden');
    
    const currentQ = appState.practiceQuestions[appState.currentQuestionIndex];
    document.getElementById('current-question').textContent = appState.currentQuestionIndex + 1;
    document.getElementById('total-questions').textContent = appState.practiceQuestions.length;

    contentDiv.innerHTML = `
        <div class="practice-card">
            <div class="mb-4">
                <span class="knowledge-tag">${currentQ.knowledgePoint}</span>
                <span class="difficulty-${currentQ.difficulty} ml-2">难度: ${getDifficultyText(currentQ.difficulty)}</span>
            </div>
            <div class="text-lg text-gray-800 mb-6">${currentQ.question}</div>
            ${currentQ.type === 'choice' ? renderChoices(currentQ.options) : renderInputAnswer()}
            <div class="mt-6 flex justify-between">
                <button onclick="previousQuestion()" ${appState.currentQuestionIndex === 0 ? 'disabled' : ''} 
                    class="px-6 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors disabled:opacity-50">
                    上一题
                </button>
                <button onclick="submitPracticeAnswer()" 
                    class="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors">
                    提交答案
                </button>
                <button onclick="nextQuestion()" ${appState.currentQuestionIndex === appState.practiceQuestions.length - 1 ? 'disabled' : ''} 
                    class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50">
                    下一题
                </button>
            </div>
            <div id="practice-feedback" class="mt-4"></div>
        </div>
    `;
}

function renderChoices(options) {
    return `
        <div class="space-y-3">
            ${options.map((opt, idx) => `
                <div class="answer-option" onclick="selectOption(${idx})">
                    <span class="font-bold">${String.fromCharCode(65 + idx)}. </span>${opt}
                </div>
            `).join('')}
        </div>
    `;
}

function renderInputAnswer() {
    return `
        <div>
            <label class="block text-gray-700 font-bold mb-2">你的答案：</label>
            <input type="text" id="practice-input-answer" 
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="请输入你的答案...">
        </div>
    `;
}

window.selectOption = function(index) {
    document.querySelectorAll('.answer-option').forEach((opt, idx) => {
        opt.classList.remove('selected');
        if (idx === index) {
            opt.classList.add('selected');
        }
    });
};

window.submitPracticeAnswer = function() {
    const currentQ = appState.practiceQuestions[appState.currentQuestionIndex];
    let userAnswer;

    if (currentQ.type === 'choice') {
        const selected = document.querySelector('.answer-option.selected');
        if (!selected) {
            showEncouragement('请先选择一个答案哦！😊');
            return;
        }
        userAnswer = Array.from(document.querySelectorAll('.answer-option')).indexOf(selected);
    } else {
        userAnswer = document.getElementById('practice-input-answer').value.trim();
        if (!userAnswer) {
            showEncouragement('请输入你的答案哦！✍️');
            return;
        }
    }

    const generator = new PracticeGenerator();
    const result = generator.checkAnswer(currentQ, userAnswer);
    
    const feedbackDiv = document.getElementById('practice-feedback');
    feedbackDiv.innerHTML = result.feedback;

    if (result.correct) {
        showEncouragement('答对了！你真棒！🎉');
        appState.userProgress.problemsSolved++;
    } else {
        showEncouragement('再想想，你一定能做对的！💪');
    }

    updateProgressDisplay();
};

window.previousQuestion = function() {
    if (appState.currentQuestionIndex > 0) {
        appState.currentQuestionIndex--;
        displayPracticeQuestion();
    }
};

window.nextQuestion = function() {
    if (appState.currentQuestionIndex < appState.practiceQuestions.length - 1) {
        appState.currentQuestionIndex++;
        displayPracticeQuestion();
    }
};

function getDifficultyText(difficulty) {
    const map = {
        'easy': '简单',
        'medium': '中等',
        'hard': '困难'
    };
    return map[difficulty] || '中等';
}

// 进度追踪模块初始化
function initProgressSection() {
    updateProgressDisplay();
}

function updateProgressDisplay() {
    document.getElementById('concepts-learned').textContent = appState.userProgress.conceptsLearned;
    document.getElementById('problems-solved').textContent = appState.userProgress.problemsSolved;
    document.getElementById('study-time').textContent = appState.userProgress.studyTime;
    document.getElementById('accuracy-rate').textContent = appState.userProgress.accuracyRate + '%';
}

function updateProgressCharts() {
    const chartManager = new ChartManager();
    
    // 知识点掌握情况雷达图
    chartManager.createKnowledgeChart('knowledge-chart', appState.userProgress.knowledgePoints);
    
    // 每日练习统计柱状图
    chartManager.createDailyChart('daily-chart', appState.userProgress.dailyPractice);
}

// 鼓励提示功能
function showEncouragement(message) {
    const toast = document.getElementById('encouragement-toast');
    const textSpan = document.getElementById('encouragement-text');
    
    textSpan.textContent = message;
    toast.classList.remove('hidden');
    toast.classList.add('show-toast');

    setTimeout(() => {
        toast.classList.remove('show-toast');
        toast.classList.add('hide-toast');
        setTimeout(() => {
            toast.classList.add('hidden');
            toast.classList.remove('hide-toast');
        }, 500);
    }, 3000);
}

// 导出全局函数供HTML使用
window.showEncouragement = showEncouragement;

// 学习时间追踪
let studyStartTime = Date.now();
setInterval(() => {
    const elapsed = Math.floor((Date.now() - studyStartTime) / 60000);
    appState.userProgress.studyTime = 45 + elapsed;
    if (document.getElementById('study-time')) {
        document.getElementById('study-time').textContent = appState.userProgress.studyTime;
    }
}, 60000);

// 交通计算器功能
let transportCalculator = new TransportCalculator();

// 更新公式显示
function updateFormulas() {
    const planABase = parseFloat(document.getElementById('planA-base').value);
    const planARate = parseFloat(document.getElementById('planA-rate').value);
    const planBBase = parseFloat(document.getElementById('planB-base').value);
    const planBRate = parseFloat(document.getElementById('planB-rate').value);

    transportCalculator.updatePlan('A', planABase, planARate);
    transportCalculator.updatePlan('B', planBBase, planBRate);

    document.getElementById('planA-formula').textContent = transportCalculator.getFormula('A');
    document.getElementById('planB-formula').textContent = transportCalculator.getFormula('B');
}

// 监听输入变化
document.addEventListener('DOMContentLoaded', () => {
    const inputs = ['planA-base', 'planA-rate', 'planB-base', 'planB-rate'];
    inputs.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('input', updateFormulas);
        }
    });
});

// 计算并显示结果
window.calculateTransport = function() {
    updateFormulas();

    // 显示结果区域
    document.getElementById('calculator-results').classList.remove('hidden');

    // 生成图表
    transportCalculator.createComparisonChart('transport-chart');

    // 生成交点分析
    document.getElementById('intersection-analysis').innerHTML = transportCalculator.generateIntersectionAnalysis();

    // 生成对比表格
    document.getElementById('comparison-table-body').innerHTML = transportCalculator.generateComparisonTable();

    // 生成决策建议
    document.getElementById('decision-advice').innerHTML = transportCalculator.generateDecisionAdvice();

    // 滚动到结果区域
    document.getElementById('calculator-results').scrollIntoView({ behavior: 'smooth', block: 'start' });

    showEncouragement('太棒了！你已经学会用一次函数分析实际问题了！🎉');
};

// 加载预设场景
window.loadScenario = function(scenario) {
    const scenarios = {
        'taxi-bus': {
            planA: { base: 10, rate: 2.4, name: '出租车' },
            planB: { base: 2, rate: 0.5, name: '公交车' }
        },
        'taxi-subway': {
            planA: { base: 10, rate: 2.4, name: '出租车' },
            planB: { base: 3, rate: 0.4, name: '地铁' }
        },
        'bike-bus': {
            planA: { base: 1.5, rate: 0.5, name: '共享单车' },
            planB: { base: 2, rate: 0.3, name: '公交车' }
        }
    };

    const selected = scenarios[scenario];
    if (selected) {
        document.getElementById('planA-base').value = selected.planA.base;
        document.getElementById('planA-rate').value = selected.planA.rate;
        document.getElementById('planB-base').value = selected.planB.base;
        document.getElementById('planB-rate').value = selected.planB.rate;

        updateFormulas();
        showEncouragement(`已加载场景：${selected.planA.name} vs ${selected.planB.name}！点击计算按钮查看分析结果！`);
    }
};