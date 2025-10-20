// 概念讲解模块
export class ConceptExplainer {
    constructor() {
        this.concepts = {
            basic: this.getBasicConcept(),
            slope: this.getSlopeConcept(),
            intercept: this.getInterceptConcept(),
            graph: this.getGraphConcept(),
            properties: this.getPropertiesConcept(),
            application: this.getApplicationConcept()
        };
    }

    getExplanation(concept) {
        return this.concepts[concept] || this.concepts.basic;
    }

    getBasicConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-blue-100 to-indigo-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-indigo-800 mb-4">📖 什么是一次函数？</h3>
                    <p class="text-lg text-gray-700 mb-4">
                        一次函数是形如 <span class="math-formula">y = kx + b</span> 的函数，其中 k 和 b 都是常数，且 k ≠ 0。
                    </p>
                    <div class="bg-white rounded-lg p-4 mt-4">
                        <p class="text-gray-700"><strong>通俗理解：</strong></p>
                        <p class="text-gray-700 mt-2">
                            想象你坐出租车，车费 = 起步价 + 每公里价格 × 公里数
                        </p>
                        <p class="text-gray-700 mt-2">
                            如果起步价是10元，每公里2元，那么：<br>
                            <span class="math-formula">车费 = 2 × 公里数 + 10</span>
                        </p>
                        <p class="text-gray-700 mt-2">
                            这就是一个一次函数！其中 k=2（每公里价格），b=10（起步价）
                        </p>
                    </div>
                </div>

                <div class="image-container">
                    <img src="https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/7b406dbf-f040-4aa7-93f7-89f30bcbf97a/image_1760871695_1_1.jpg" 
                         alt="出租车计价器" class="max-h-64">
                    <p class="text-sm text-gray-600 mt-2">出租车计价器就是一次函数的实际应用</p>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">🔑 关键要素</h4>
                    <ul class="space-y-2 text-gray-700">
                        <li><strong>k（斜率）：</strong>表示变化率，决定直线的倾斜程度</li>
                        <li><strong>b（截距）：</strong>表示初始值，决定直线与y轴的交点</li>
                        <li><strong>x（自变量）：</strong>可以自由取值的量</li>
                        <li><strong>y（因变量）：</strong>随x变化而变化的量</li>
                    </ul>
                </div>

                <div class="bg-green-50 rounded-xl p-6">
                    <h4 class="text-lg font-bold text-green-800 mb-3">✨ 生活中的例子</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-white rounded-lg p-4">
                            <div class="text-2xl mb-2">🚕</div>
                            <strong>出租车计费</strong>
                            <p class="text-sm text-gray-600 mt-1">费用 = 单价 × 里程 + 起步价</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <div class="text-2xl mb-2">📱</div>
                            <strong>手机话费</strong>
                            <p class="text-sm text-gray-600 mt-1">话费 = 每分钟费用 × 通话时长 + 月租</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <div class="text-2xl mb-2">💧</div>
                            <strong>水费计算</strong>
                            <p class="text-sm text-gray-600 mt-1">水费 = 单价 × 用水量 + 基本费</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <div class="text-2xl mb-2">🏃</div>
                            <strong>匀速运动</strong>
                            <p class="text-sm text-gray-600 mt-1">路程 = 速度 × 时间 + 初始位置</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getSlopeConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-purple-100 to-pink-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-purple-800 mb-4">📐 斜率k的意义</h3>
                    <p class="text-lg text-gray-700 mb-4">
                        斜率k表示<span class="highlight">变化率</span>，即y随x每变化1个单位时的变化量。
                    </p>
                    <div class="math-formula text-center">
                        k = (y₂ - y₁) / (x₂ - x₁)
                    </div>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-purple-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">🚗 出租车例子详解</h4>
                    <p class="text-gray-700 mb-3">
                        假设出租车每公里收费2元，起步价10元：
                    </p>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>公里数(x)</th>
                                <th>车费(y)</th>
                                <th>变化量</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>0公里</td>
                                <td>10元</td>
                                <td>-</td>
                            </tr>
                            <tr>
                                <td>1公里</td>
                                <td>12元</td>
                                <td>+2元</td>
                            </tr>
                            <tr>
                                <td>2公里</td>
                                <td>14元</td>
                                <td>+2元</td>
                            </tr>
                            <tr>
                                <td>3公里</td>
                                <td>16元</td>
                                <td>+2元</td>
                            </tr>
                        </tbody>
                    </table>
                    <p class="text-gray-700 mt-4">
                        可以看到，每增加1公里，车费就增加2元，这个<strong>2</strong>就是斜率k！
                    </p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-green-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-green-800 mb-3">✅ k > 0（正斜率）</h4>
                        <ul class="space-y-2 text-gray-700">
                            <li>• 函数递增（从左到右上升）</li>
                            <li>• y随x增大而增大</li>
                            <li>• k越大，直线越陡</li>
                            <li>• 例：收入随工作时间增加</li>
                        </ul>
                        <div class="mt-4 text-center">
                            <div class="inline-block bg-white p-4 rounded-lg">
                                <svg width="150" height="100" viewBox="0 0 150 100">
                                    <line x1="10" y1="90" x2="140" y2="90" stroke="#666" stroke-width="2"/>
                                    <line x1="10" y1="90" x2="10" y2="10" stroke="#666" stroke-width="2"/>
                                    <line x1="10" y1="80" x2="140" y2="20" stroke="#10B981" stroke-width="3"/>
                                    <text x="145" y="95" font-size="12">x</text>
                                    <text x="5" y="8" font-size="12">y</text>
                                </svg>
                            </div>
                        </div>
                    </div>

                    <div class="bg-red-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-red-800 mb-3">⚠️ k < 0（负斜率）</h4>
                        <ul class="space-y-2 text-gray-700">
                            <li>• 函数递减（从左到右下降）</li>
                            <li>• y随x增大而减小</li>
                            <li>• |k|越大，直线越陡</li>
                            <li>• 例：水箱水量随放水时间减少</li>
                        </ul>
                        <div class="mt-4 text-center">
                            <div class="inline-block bg-white p-4 rounded-lg">
                                <svg width="150" height="100" viewBox="0 0 150 100">
                                    <line x1="10" y1="90" x2="140" y2="90" stroke="#666" stroke-width="2"/>
                                    <line x1="10" y1="90" x2="10" y2="10" stroke="#666" stroke-width="2"/>
                                    <line x1="10" y1="20" x2="140" y2="80" stroke="#EF4444" stroke-width="3"/>
                                    <text x="145" y="95" font-size="12">x</text>
                                    <text x="5" y="8" font-size="12">y</text>
                                </svg>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="bg-blue-50 border-l-4 border-blue-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-blue-800 mb-3">💡 记忆技巧</h4>
                    <p class="text-gray-700">
                        <strong>斜率k就像爬山的坡度：</strong><br>
                        • k越大，坡越陡，爬得越累（变化越快）<br>
                        • k为正，向上爬（增长）<br>
                        • k为负，向下走（下降）<br>
                        • k的绝对值越大，坡度越大
                    </p>
                </div>
            </div>
        `;
    }

    getInterceptConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-green-100 to-teal-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-green-800 mb-4">📍 截距b的意义</h3>
                    <p class="text-lg text-gray-700 mb-4">
                        截距b表示<span class="highlight">初始值</span>，即当x=0时，y的值。
                    </p>
                    <div class="math-formula text-center">
                        当 x = 0 时，y = b
                    </div>
                    <p class="text-gray-700 mt-4 text-center">
                        在坐标系中，b是直线与y轴交点的纵坐标
                    </p>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-green-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">🚕 继续出租车的例子</h4>
                    <p class="text-gray-700 mb-4">
                        车费 = 2 × 公里数 + 10，这里的<strong>10元</strong>就是截距b
                    </p>
                    <div class="bg-yellow-50 rounded-lg p-4">
                        <p class="text-gray-700">
                            <strong>思考：</strong>为什么叫"起步价"？<br>
                            因为即使你一公里都不走（x=0），也要付10元！<br>
                            这就是截距b的实际意义——<span class="highlight">初始状态的值</span>
                        </p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="bg-blue-50 rounded-xl p-6 text-center">
                        <h4 class="text-lg font-bold text-blue-800 mb-3">b > 0</h4>
                        <p class="text-gray-700 mb-3">直线与y轴交于正半轴</p>
                        <div class="bg-white p-4 rounded-lg inline-block">
                            <svg width="120" height="120" viewBox="0 0 120 120">
                                <line x1="10" y1="110" x2="110" y2="110" stroke="#666" stroke-width="2"/>
                                <line x1="10" y1="110" x2="10" y2="10" stroke="#666" stroke-width="2"/>
                                <line x1="10" y1="70" x2="110" y2="20" stroke="#3B82F6" stroke-width="3"/>
                                <circle cx="10" cy="70" r="4" fill="#EF4444"/>
                                <text x="15" y="75" font-size="12" fill="#EF4444">b</text>
                            </svg>
                        </div>
                    </div>

                    <div class="bg-purple-50 rounded-xl p-6 text-center">
                        <h4 class="text-lg font-bold text-purple-800 mb-3">b = 0</h4>
                        <p class="text-gray-700 mb-3">直线过原点</p>
                        <div class="bg-white p-4 rounded-lg inline-block">
                            <svg width="120" height="120" viewBox="0 0 120 120">
                                <line x1="10" y1="110" x2="110" y2="110" stroke="#666" stroke-width="2"/>
                                <line x1="10" y1="110" x2="10" y2="10" stroke="#666" stroke-width="2"/>
                                <line x1="10" y1="110" x2="110" y2="10" stroke="#8B5CF6" stroke-width="3"/>
                                <circle cx="10" cy="110" r="4" fill="#EF4444"/>
                                <text x="15" y="115" font-size="12" fill="#EF4444">0</text>
                            </svg>
                        </div>
                    </div>

                    <div class="bg-orange-50 rounded-xl p-6 text-center">
                        <h4 class="text-lg font-bold text-orange-800 mb-3">b < 0</h4>
                        <p class="text-gray-700 mb-3">直线与y轴交于负半轴</p>
                        <div class="bg-white p-4 rounded-lg inline-block">
                            <svg width="120" height="120" viewBox="0 0 120 120">
                                <line x1="10" y1="60" x2="110" y2="60" stroke="#666" stroke-width="2"/>
                                <line x1="10" y1="110" x2="10" y2="10" stroke="#666" stroke-width="2"/>
                                <line x1="10" y1="80" x2="110" y2="30" stroke="#F97316" stroke-width="3"/>
                                <circle cx="10" cy="80" r="4" fill="#EF4444"/>
                                <text x="15" y="85" font-size="12" fill="#EF4444">b</text>
                            </svg>
                        </div>
                    </div>
                </div>

                <div class="bg-indigo-50 rounded-xl p-6">
                    <h4 class="text-lg font-bold text-indigo-800 mb-4">🌟 更多生活例子</h4>
                    <div class="space-y-3">
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-indigo-700">📱 手机套餐：</strong>
                            <p class="text-gray-700 mt-1">
                                话费 = 0.1 × 通话分钟数 + 30<br>
                                截距b=30元，表示月租费（即使不打电话也要付）
                            </p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-indigo-700">🏊 游泳池放水：</strong>
                            <p class="text-gray-700 mt-1">
                                水量 = -5 × 时间 + 100<br>
                                截距b=100吨，表示初始水量
                            </p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-indigo-700">🌡️ 温度变化：</strong>
                            <p class="text-gray-700 mt-1">
                                温度 = 2 × 小时数 + 15<br>
                                截距b=15℃，表示初始温度
                            </p>
                        </div>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">💡 记忆技巧</h4>
                    <p class="text-gray-700">
                        <strong>截距b就像起跑线：</strong><br>
                        • 比赛开始前（x=0），你已经在起跑线上了<br>
                        • b>0：起跑线在前面（有优势）<br>
                        • b=0：从原点出发（公平竞争）<br>
                        • b<0：起跑线在后面（需要追赶）
                    </p>
                </div>
            </div>
        `;
    }

    getGraphConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-orange-100 to-red-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-orange-800 mb-4">📈 一次函数的图像</h3>
                    <p class="text-lg text-gray-700 mb-4">
                        一次函数 y = kx + b 的图像是一条<span class="highlight">直线</span>
                    </p>
                </div>

                <div class="image-container">
                    <img src="https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/764f999d-1660-44d6-9c95-0bc28e5b50fd/image_1760871687_1_1.png" 
                         alt="一次函数图像" class="max-h-80">
                    <p class="text-sm text-gray-600 mt-2">一次函数在坐标系中的图像</p>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-orange-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">✏️ 如何画一次函数图像？</h4>
                    <div class="space-y-4">
                        <div class="solution-step">
                            <span class="solution-step-number">1</span>
                            <div class="solution-step-content">
                                <strong>列表取点</strong><br>
                                选择几个x的值，计算对应的y值，列成表格
                            </div>
                        </div>
                        <div class="solution-step">
                            <span class="solution-step-number">2</span>
                            <div class="solution-step-content">
                                <strong>描点</strong><br>
                                在坐标系中标出这些点的位置
                            </div>
                        </div>
                        <div class="solution-step">
                            <span class="solution-step-number">3</span>
                            <div class="solution-step-content">
                                <strong>连线</strong><br>
                                用直尺把这些点连成一条直线
                            </div>
                        </div>
                    </div>
                </div>

                <div class="bg-blue-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-blue-800 mb-4">🎯 实战演练：画出 y = 2x + 1</h4>
                    <div class="bg-white rounded-lg p-4 mb-4">
                        <strong>步骤1：列表</strong>
                        <table class="data-table mt-3">
                            <thead>
                                <tr>
                                    <th>x</th>
                                    <th>-2</th>
                                    <th>-1</th>
                                    <th>0</th>
                                    <th>1</th>
                                    <th>2</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>y</strong></td>
                                    <td>-3</td>
                                    <td>-1</td>
                                    <td>1</td>
                                    <td>3</td>
                                    <td>5</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="bg-white rounded-lg p-4">
                        <strong>步骤2和3：描点连线</strong>
                        <div class="mt-3 text-center">
                            <svg width="300" height="300" viewBox="0 0 300 300" class="inline-block bg-white">
                                <!-- 坐标轴 -->
                                <line x1="30" y1="270" x2="270" y2="270" stroke="#666" stroke-width="2"/>
                                <line x1="150" y1="30" x2="150" y2="270" stroke="#666" stroke-width="2"/>
                                <!-- 刻度 -->
                                <text x="145" y="285" font-size="12">0</text>
                                <text x="185" y="285" font-size="12">1</text>
                                <text x="225" y="285" font-size="12">2</text>
                                <text x="105" y="285" font-size="12">-1</text>
                                <text x="65" y="285" font-size="12">-2</text>
                                <text x="155" y="235" font-size="12">1</text>
                                <text x="155" y="195" font-size="12">2</text>
                                <text x="155" y="155" font-size="12">3</text>
                                <!-- 函数直线 -->
                                <line x1="70" y1="210" x2="230" y2="90" stroke="#3B82F6" stroke-width="3"/>
                                <!-- 点 -->
                                <circle cx="70" cy="210" r="4" fill="#EF4444"/>
                                <circle cx="110" cy="190" r="4" fill="#EF4444"/>
                                <circle cx="150" cy="170" r="4" fill="#EF4444"/>
                                <circle cx="190" cy="150" r="4" fill="#EF4444"/>
                                <circle cx="230" cy="130" r="4" fill="#EF4444"/>
                                <!-- 标签 -->
                                <text x="240" y="85" font-size="14" fill="#3B82F6" font-weight="bold">y=2x+1</text>
                            </svg>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-green-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-green-800 mb-3">✨ 快速画图技巧</h4>
                        <p class="text-gray-700 mb-3">
                            <strong>两点法：</strong>因为两点确定一条直线，所以只需要找两个点！
                        </p>
                        <ul class="space-y-2 text-gray-700">
                            <li><strong>方法1：</strong>找x=0和y=0的点</li>
                            <li><strong>方法2：</strong>找两个容易计算的x值</li>
                            <li><strong>建议：</strong>多找一个点验证</li>
                        </ul>
                    </div>

                    <div class="bg-purple-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-purple-800 mb-3">⚠️ 注意事项</h4>
                        <ul class="space-y-2 text-gray-700">
                            <li>• 用直尺画线，保证是直线</li>
                            <li>• 箭头表示直线延伸到无穷</li>
                            <li>• 标注函数表达式</li>
                            <li>• 坐标轴要标清楚</li>
                        </ul>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">💡 记忆口诀</h4>
                    <p class="text-gray-700 text-lg">
                        <strong>一次函数是直线，两点确定最方便<br>
                        列表描点再连线，箭头两端要画全</strong>
                    </p>
                </div>
            </div>
        `;
    }

    getPropertiesConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-pink-100 to-purple-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-pink-800 mb-4">🔍 一次函数的性质</h3>
                    <p class="text-lg text-gray-700">
                        掌握一次函数的性质，可以快速判断函数的特征和图像位置
                    </p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6">
                        <h4 class="text-xl font-bold text-green-800 mb-4">📈 增减性</h4>
                        <div class="space-y-4">
                            <div class="bg-white rounded-lg p-4">
                                <strong class="text-green-700">当 k > 0 时：</strong>
                                <p class="text-gray-700 mt-2">
                                    • 函数单调递增<br>
                                    • y随x增大而增大<br>
                                    • 图像从左下到右上
                                </p>
                                <div class="mt-3 text-center">
                                    <span class="text-4xl">📈</span>
                                </div>
                            </div>
                            <div class="bg-white rounded-lg p-4">
                                <strong class="text-red-700">当 k < 0 时：</strong>
                                <p class="text-gray-700 mt-2">
                                    • 函数单调递减<br>
                                    • y随x增大而减小<br>
                                    • 图像从左上到右下
                                </p>
                                <div class="mt-3 text-center">
                                    <span class="text-4xl">📉</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6">
                        <h4 class="text-xl font-bold text-blue-800 mb-4">📍 图像位置</h4>
                        <div class="bg-white rounded-lg p-4 mb-3">
                            <strong class="text-blue-700">由k和b共同决定：</strong>
                            <table class="data-table mt-3">
                                <thead>
                                    <tr>
                                        <th>k和b的符号</th>
                                        <th>经过象限</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>k>0, b>0</td>
                                        <td>一、二、三</td>
                                    </tr>
                                    <tr>
                                        <td>k>0, b<0</td>
                                        <td>一、三、四</td>
                                    </tr>
                                    <tr>
                                        <td>k<0, b>0</td>
                                        <td>一、二、四</td>
                                    </tr>
                                    <tr>
                                        <td>k<0, b<0</td>
                                        <td>二、三、四</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-purple-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">🎨 四种基本图像</h4>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div class="text-center">
                            <div class="bg-gray-50 p-3 rounded-lg mb-2">
                                <svg width="100" height="100" viewBox="0 0 100 100">
                                    <line x1="10" y1="90" x2="90" y2="90" stroke="#ccc" stroke-width="1"/>
                                    <line x1="50" y1="10" x2="50" y2="90" stroke="#ccc" stroke-width="1"/>
                                    <line x1="20" y1="70" x2="80" y2="30" stroke="#10B981" stroke-width="2"/>
                                </svg>
                            </div>
                            <p class="text-sm font-bold text-green-700">k>0, b>0</p>
                            <p class="text-xs text-gray-600">一、二、三象限</p>
                        </div>
                        <div class="text-center">
                            <div class="bg-gray-50 p-3 rounded-lg mb-2">
                                <svg width="100" height="100" viewBox="0 0 100 100">
                                    <line x1="10" y1="50" x2="90" y2="50" stroke="#ccc" stroke-width="1"/>
                                    <line x1="50" y1="10" x2="50" y2="90" stroke="#ccc" stroke-width="1"/>
                                    <line x1="20" y1="80" x2="80" y2="20" stroke="#3B82F6" stroke-width="2"/>
                                </svg>
                            </div>
                            <p class="text-sm font-bold text-blue-700">k>0, b<0</p>
                            <p class="text-xs text-gray-600">一、三、四象限</p>
                        </div>
                        <div class="text-center">
                            <div class="bg-gray-50 p-3 rounded-lg mb-2">
                                <svg width="100" height="100" viewBox="0 0 100 100">
                                    <line x1="10" y1="90" x2="90" y2="90" stroke="#ccc" stroke-width="1"/>
                                    <line x1="50" y1="10" x2="50" y2="90" stroke="#ccc" stroke-width="1"/>
                                    <line x1="20" y1="30" x2="80" y2="70" stroke="#F59E0B" stroke-width="2"/>
                                </svg>
                            </div>
                            <p class="text-sm font-bold text-orange-700">k<0, b>0</p>
                            <p class="text-xs text-gray-600">一、二、四象限</p>
                        </div>
                        <div class="text-center">
                            <div class="bg-gray-50 p-3 rounded-lg mb-2">
                                <svg width="100" height="100" viewBox="0 0 100 100">
                                    <line x1="10" y1="50" x2="90" y2="50" stroke="#ccc" stroke-width="1"/>
                                    <line x1="50" y1="10" x2="50" y2="90" stroke="#ccc" stroke-width="1"/>
                                    <line x1="20" y1="20" x2="80" y2="80" stroke="#EF4444" stroke-width="2"/>
                                </svg>
                            </div>
                            <p class="text-sm font-bold text-red-700">k<0, b<0</p>
                            <p class="text-xs text-gray-600">二、三、四象限</p>
                        </div>
                    </div>
                </div>

                <div class="bg-indigo-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-indigo-800 mb-4">🎯 特殊情况</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-indigo-700">当 b = 0 时：</strong>
                            <p class="text-gray-700 mt-2">
                                y = kx（正比例函数）<br>
                                • 图像过原点<br>
                                • k>0时过一、三象限<br>
                                • k<0时过二、四象限
                            </p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-indigo-700">平行与垂直：</strong>
                            <p class="text-gray-700 mt-2">
                                • 两直线平行 ⟺ k₁ = k₂<br>
                                • 两直线垂直 ⟺ k₁ × k₂ = -1<br>
                                （例：k₁=2, k₂=-1/2）
                            </p>
                        </div>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">💡 判断技巧</h4>
                    <p class="text-gray-700">
                        <strong>看图像快速判断k和b：</strong><br>
                        1. 看倾斜方向判断k：上升为正，下降为负<br>
                        2. 看与y轴交点判断b：交点在上方为正，下方为负<br>
                        3. 看倾斜程度判断|k|：越陡|k|越大
                    </p>
                </div>
            </div>
        `;
    }

    getApplicationConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-yellow-100 to-orange-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-orange-800 mb-4">🚀 一次函数的实际应用</h3>
                    <p class="text-lg text-gray-700">
                        一次函数在生活中无处不在，让我们看看如何用它解决实际问题！
                    </p>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-orange-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">🚕 经典案例：出行方案选择</h4>
                    <div class="bg-blue-50 rounded-lg p-4 mb-4">
                        <strong>问题情境：</strong>
                        <p class="text-gray-700 mt-2">
                            小明要去10公里外的地方，有两种出行方式：<br>
                            <strong>方案A（出租车）：</strong>起步价10元，每公里2元<br>
                            <strong>方案B（网约车）：</strong>起步价8元，每公里2.5元<br>
                            <strong>问题：</strong>如何选择更省钱？
                        </p>
                    </div>

                    <div class="space-y-4">
                        <div class="solution-step">
                            <span class="solution-step-number">1</span>
                            <div class="solution-step-content">
                                <strong>建立函数模型</strong><br>
                                设路程为x公里，费用为y元<br>
                                方案A：y₁ = 2x + 10<br>
                                方案B：y₂ = 2.5x + 8
                            </div>
                        </div>

                        <div class="solution-step">
                            <span class="solution-step-number">2</span>
                            <div class="solution-step-content">
                                <strong>找出分界点</strong><br>
                                令 y₁ = y₂<br>
                                2x + 10 = 2.5x + 8<br>
                                解得：x = 4公里
                            </div>
                        </div>

                        <div class="solution-step">
                            <span class="solution-step-number">3</span>
                            <div class="solution-step-content">
                                <strong>得出结论</strong><br>
                                • 当路程 < 4公里时，选方案B更省钱<br>
                                • 当路程 = 4公里时，两方案费用相同<br>
                                • 当路程 > 4公里时，选方案A更省钱<br>
                                <strong class="text-green-700">所以10公里选方案A！</strong>
                            </div>
                        </div>
                    </div>

                    <div class="mt-4 bg-gray-50 rounded-lg p-4">
                        <strong>图像分析：</strong>
                        <div class="text-center mt-3">
                            <svg width="400" height="250" viewBox="0 0 400 250" class="inline-block bg-white rounded">
                                <line x1="40" y1="210" x2="380" y2="210" stroke="#666" stroke-width="2"/>
                                <line x1="40" y1="210" x2="40" y2="30" stroke="#666" stroke-width="2"/>
                                <line x1="40" y1="170" x2="380" y2="10" stroke="#3B82F6" stroke-width="3"/>
                                <line x1="40" y1="190" x2="380" y2="20" stroke="#EF4444" stroke-width="3"/>
                                <circle cx="160" cy="130" r="5" fill="#10B981"/>
                                <text x="50" y="25" font-size="14">费用(元)</text>
                                <text x="360" y="230" font-size="14">路程(km)</text>
                                <text x="200" y="120" font-size="14" fill="#10B981" font-weight="bold">交点(4,18)</text>
                                <text x="300" y="30" font-size="14" fill="#3B82F6">方案A</text>
                                <text x="300" y="50" font-size="14" fill="#EF4444">方案B</text>
                            </svg>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-green-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-green-800 mb-3">📱 案例2：手机套餐</h4>
                        <p class="text-gray-700 mb-3">
                            <strong>套餐A：</strong>月租30元，每分钟0.1元<br>
                            <strong>套餐B：</strong>月租50元，每分钟0.05元
                        </p>
                        <div class="bg-white rounded-lg p-3">
                            <strong>建模：</strong><br>
                            A: y = 0.1x + 30<br>
                            B: y = 0.05x + 50<br>
                            <strong class="text-green-700">分界点：400分钟</strong>
                        </div>
                    </div>

                    <div class="bg-purple-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-purple-800 mb-3">💧 案例3：水费计算</h4>
                        <p class="text-gray-700 mb-3">
                            基本费5元，每吨水费3元<br>
                            用水x吨，总费用y元
                        </p>
                        <div class="bg-white rounded-lg p-3">
                            <strong>建模：</strong><br>
                            y = 3x + 5<br>
                            <strong class="text-purple-700">可预测任意用水量的费用</strong>
                        </div>
                    </div>
                </div>

                <div class="bg-indigo-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-indigo-800 mb-4">🎯 解决实际问题的步骤</h4>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div class="bg-white rounded-lg p-4 text-center">
                            <div class="text-3xl mb-2">1️⃣</div>
                            <strong>审题</strong>
                            <p class="text-sm text-gray-600 mt-1">找出变量和常量</p>
                        </div>
                        <div class="bg-white rounded-lg p-4 text-center">
                            <div class="text-3xl mb-2">2️⃣</div>
                            <strong>建模</strong>
                            <p class="text-sm text-gray-600 mt-1">列出函数表达式</p>
                        </div>
                        <div class="bg-white rounded-lg p-4 text-center">
                            <div class="text-3xl mb-2">3️⃣</div>
                            <strong>求解</strong>
                            <p class="text-sm text-gray-600 mt-1">计算或画图分析</p>
                        </div>
                        <div class="bg-white rounded-lg p-4 text-center">
                            <div class="text-3xl mb-2">4️⃣</div>
                            <strong>检验</strong>
                            <p class="text-sm text-gray-600 mt-1">验证答案合理性</p>
                        </div>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">💡 应用技巧</h4>
                    <ul class="space-y-2 text-gray-700">
                        <li><strong>1. 识别关键词：</strong>"起步价"→截距b，"每...多少"→斜率k</li>
                        <li><strong>2. 画图辅助：</strong>复杂问题用图像更直观</li>
                        <li><strong>3. 找交点：</strong>比较方案时，交点是决策的关键</li>
                        <li><strong>4. 检验实际意义：</strong>答案要符合实际情况（如距离不能为负）</li>
                    </ul>
                </div>
            </div>
        `;
    }

    generateCustomExplanation(question) {
        // 根据问题关键词生成相应讲解
        const keywords = {
            '斜率': 'slope',
            '截距': 'intercept',
            '图像': 'graph',
            '性质': 'properties',
            '应用': 'application',
            '画': 'graph',
            '增减': 'properties',
            '象限': 'properties'
        };

        for (let [keyword, concept] of Object.entries(keywords)) {
            if (question.includes(keyword)) {
                return this.concepts[concept];
            }
        }

        // 默认返回基础概念
        return `
            <div class="bg-blue-50 rounded-xl p-6">
                <h3 class="text-xl font-bold text-blue-800 mb-4">关于"${question}"的解答</h3>
                <p class="text-gray-700 mb-4">
                    这是一个很好的问题！让我来帮你理解：
                </p>
                <div class="bg-white rounded-lg p-4">
                    <p class="text-gray-700">
                        一次函数 y = kx + b 是初中数学的重要内容。理解它的关键是：
                    </p>
                    <ul class="mt-3 space-y-2 text-gray-700">
                        <li>• <strong>k（斜率）</strong>控制直线的倾斜程度和方向</li>
                        <li>• <strong>b（截距）</strong>决定直线与y轴的交点位置</li>
                        <li>• 图像是一条<strong>直线</strong></li>
                        <li>• 在生活中有广泛应用，如出租车计费、手机套餐等</li>
                    </ul>
                </div>
                <div class="mt-4 bg-yellow-50 rounded-lg p-4">
                    <p class="text-gray-700">
                        💡 <strong>建议：</strong>你可以点击上方的具体知识点按钮，获取更详细的讲解。
                        如果还有疑问，欢迎继续提问！
                    </p>
                </div>
            </div>
        `;
    }
}