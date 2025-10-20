// 进度追踪模块
export class ProgressTracker {
    constructor() {
        this.loadProgress();
    }

    loadProgress() {
        // 从localStorage加载进度数据
        const saved = localStorage.getItem('mathLearningProgress');
        if (saved) {
            this.progress = JSON.parse(saved);
        } else {
            this.progress = this.initProgress();
        }
    }

    initProgress() {
        return {
            conceptsLearned: 0,
            problemsSolved: 0,
            studyTime: 0,
            accuracyRate: 0,
            totalAttempts: 0,
            correctAttempts: 0,
            knowledgePoints: {
                '基础概念': 0,
                '斜率理解': 0,
                '截距理解': 0,
                '图像绘制': 0,
                '性质分析': 0,
                '实际应用': 0
            },
            dailyPractice: [0, 0, 0, 0, 0, 0, 0], // 最近7天
            achievements: [],
            lastStudyDate: new Date().toDateString()
        };
    }

    saveProgress() {
        localStorage.setItem('mathLearningProgress', JSON.stringify(this.progress));
    }

    updateConceptLearned(conceptName) {
        this.progress.conceptsLearned++;
        
        // 更新对应知识点的掌握度
        const knowledgeMap = {
            'basic': '基础概念',
            'slope': '斜率理解',
            'intercept': '截距理解',
            'graph': '图像绘制',
            'properties': '性质分析',
            'application': '实际应用'
        };

        const knowledgePoint = knowledgeMap[conceptName];
        if (knowledgePoint && this.progress.knowledgePoints[knowledgePoint] < 100) {
            this.progress.knowledgePoints[knowledgePoint] = Math.min(
                100,
                this.progress.knowledgePoints[knowledgePoint] + 15
            );
        }

        this.checkAchievements();
        this.saveProgress();
    }

    updateProblemSolved(isCorrect, knowledgePoint) {
        this.progress.problemsSolved++;
        this.progress.totalAttempts++;
        
        if (isCorrect) {
            this.progress.correctAttempts++;
        }

        // 更新正确率
        this.progress.accuracyRate = Math.round(
            (this.progress.correctAttempts / this.progress.totalAttempts) * 100
        );

        // 更新知识点掌握度
        if (knowledgePoint && this.progress.knowledgePoints[knowledgePoint] !== undefined) {
            const change = isCorrect ? 5 : -3;
            this.progress.knowledgePoints[knowledgePoint] = Math.max(
                0,
                Math.min(100, this.progress.knowledgePoints[knowledgePoint] + change)
            );
        }

        // 更新每日练习统计
        this.updateDailyPractice();

        this.checkAchievements();
        this.saveProgress();
    }

    updateDailyPractice() {
        const today = new Date().toDateString();
        
        if (this.progress.lastStudyDate !== today) {
            // 新的一天，数组左移
            this.progress.dailyPractice.shift();
            this.progress.dailyPractice.push(1);
            this.progress.lastStudyDate = today;
        } else {
            // 同一天，增加计数
            this.progress.dailyPractice[6]++;
        }
    }

    updateStudyTime(minutes) {
        this.progress.studyTime += minutes;
        this.saveProgress();
    }

    checkAchievements() {
        const achievements = [];

        // 初学者成就
        if (this.progress.conceptsLearned >= 1 && !this.hasAchievement('beginner')) {
            achievements.push({
                id: 'beginner',
                name: '初学者',
                description: '完成首次学习',
                icon: '🌟',
                date: new Date().toLocaleDateString()
            });
        }

        // 概念大师成就
        if (this.progress.conceptsLearned >= 6 && !this.hasAchievement('concept_master')) {
            achievements.push({
                id: 'concept_master',
                name: '概念大师',
                description: '学完所有概念',
                icon: '📖',
                date: new Date().toLocaleDateString()
            });
        }

        // 解题高手成就
        if (this.progress.problemsSolved >= 50 && !this.hasAchievement('problem_solver')) {
            achievements.push({
                id: 'problem_solver',
                name: '解题高手',
                description: '解决50道题目',
                icon: '🎯',
                date: new Date().toLocaleDateString()
            });
        }

        // 数学之星成就
        if (this.progress.accuracyRate >= 95 && this.progress.totalAttempts >= 20 && !this.hasAchievement('math_star')) {
            achievements.push({
                id: 'math_star',
                name: '数学之星',
                description: '正确率达到95%',
                icon: '👑',
                date: new Date().toLocaleDateString()
            });
        }

        // 坚持学习成就
        const consecutiveDays = this.getConsecutiveDays();
        if (consecutiveDays >= 7 && !this.hasAchievement('persistent')) {
            achievements.push({
                id: 'persistent',
                name: '坚持不懈',
                description: '连续学习7天',
                icon: '🔥',
                date: new Date().toLocaleDateString()
            });
        }

        // 添加新成就
        achievements.forEach(achievement => {
            this.progress.achievements.push(achievement);
        });

        return achievements;
    }

    hasAchievement(achievementId) {
        return this.progress.achievements.some(a => a.id === achievementId);
    }

    getConsecutiveDays() {
        let count = 0;
        for (let i = this.progress.dailyPractice.length - 1; i >= 0; i--) {
            if (this.progress.dailyPractice[i] > 0) {
                count++;
            } else {
                break;
            }
        }
        return count;
    }

    getProgress() {
        return this.progress;
    }

    getKnowledgePointMastery(point) {
        return this.progress.knowledgePoints[point] || 0;
    }

    getAllKnowledgePoints() {
        return this.progress.knowledgePoints;
    }

    getDailyPracticeData() {
        return this.progress.dailyPractice;
    }

    getStatistics() {
        return {
            conceptsLearned: this.progress.conceptsLearned,
            problemsSolved: this.progress.problemsSolved,
            studyTime: this.progress.studyTime,
            accuracyRate: this.progress.accuracyRate,
            totalAttempts: this.progress.totalAttempts,
            correctAttempts: this.progress.correctAttempts
        };
    }

    resetProgress() {
        this.progress = this.initProgress();
        this.saveProgress();
    }

    exportProgress() {
        return JSON.stringify(this.progress, null, 2);
    }

    importProgress(jsonString) {
        try {
            this.progress = JSON.parse(jsonString);
            this.saveProgress();
            return true;
        } catch (e) {
            console.error('导入进度失败:', e);
            return false;
        }
    }

    // 生成学习报告
    generateReport() {
        const stats = this.getStatistics();
        const knowledgePoints = this.getAllKnowledgePoints();
        
        // 找出掌握最好和最差的知识点
        let bestPoint = { name: '', score: 0 };
        let worstPoint = { name: '', score: 100 };
        
        Object.entries(knowledgePoints).forEach(([name, score]) => {
            if (score > bestPoint.score) {
                bestPoint = { name, score };
            }
            if (score < worstPoint.score) {
                worstPoint = { name, score };
            }
        });

        return {
            summary: {
                totalStudyTime: stats.studyTime,
                problemsSolved: stats.problemsSolved,
                accuracyRate: stats.accuracyRate,
                conceptsLearned: stats.conceptsLearned
            },
            strengths: bestPoint,
            weaknesses: worstPoint,
            suggestions: this.generateSuggestions(worstPoint, stats),
            achievements: this.progress.achievements
        };
    }

    generateSuggestions(worstPoint, stats) {
        const suggestions = [];

        if (worstPoint.score < 60) {
            suggestions.push(`建议加强"${worstPoint.name}"的学习，可以多看相关概念讲解`);
        }

        if (stats.accuracyRate < 70) {
            suggestions.push('正确率还有提升空间，建议多做基础练习，打牢基础');
        }

        if (stats.problemsSolved < 20) {
            suggestions.push('练习量还不够，建议每天至少完成5道练习题');
        }

        const avgDailyPractice = this.progress.dailyPractice.reduce((a, b) => a + b, 0) / 7;
        if (avgDailyPractice < 3) {
            suggestions.push('建议保持每天学习的习惯，坚持就是胜利！');
        }

        if (suggestions.length === 0) {
            suggestions.push('你的学习状态很好，继续保持！可以尝试一些难度更高的题目');
        }

        return suggestions;
    }
}