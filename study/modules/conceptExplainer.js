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
                    <h3 class="text-2xl font-bold text-indigo-800 mb-4">📖 什么是分段函数？</h3>
                    <p class="text-lg text-gray-700 mb-4">
                        分段函数是指在<span class="highlight">不同的自变量取值范围内</span>，有<span class="highlight">不同的对应关系</span>的函数。
                    </p>
                    <div class="bg-white rounded-lg p-4 mt-4">
                        <p class="text-gray-700"><strong>通俗理解：</strong></p>
                        <p class="text-gray-700 mt-2">
                            想象你坐出租车，计费规则是这样的：
                        </p>
                        <p class="text-gray-700 mt-2">
                            • 前2公里：固定收费<span class="math-formula">10元</span>（起步价）<br>
                            • 超过2公里：每公里加收<span class="math-formula">2.7元</span>
                        </p>
                        <p class="text-gray-700 mt-3">
                            这就是一个典型的<strong>分段函数</strong>！不同距离段有不同的计费方式。
                        </p>
                        <div class="math-formula mt-4">
                            y = { 10, (x ≤ 2)<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.7x + 4.6, (x > 2) }
                        </div>
                    </div>
                </div>

                <div class="image-container">
                    <img src="https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/7b406dbf-f040-4aa7-93f7-89f30bcbf97a/image_1760871695_1_1.jpg" 
                         alt="出租车计价器" class="max-h-64">
                    <p class="text-sm text-gray-600 mt-2">出租车计价器就是分段函数的实际应用</p>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">🔑 关键要素</h4>
                    <ul class="space-y-2 text-gray-700">
                        <li><strong>分段点：</strong>函数规则发生变化的临界值（如2公里）</li>
                        <li><strong>分段区间：</strong>每个规则适用的自变量范围</li>
                        <li><strong>分段表达式：</strong>每个区间对应的函数关系式</li>
                        <li><strong>连续性：</strong>在分段点处函数值是否连续</li>
                    </ul>
                </div>

                <div class="bg-green-50 rounded-xl p-6">
                    <h4 class="text-lg font-bold text-green-800 mb-3">✨ 生活中的分段函数</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-white rounded-lg p-4">
                            <div class="text-2xl mb-2">🚕</div>
                            <strong>出租车计费</strong>
                            <p class="text-sm text-gray-600 mt-1">起步价 + 超出部分按公里计费</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <div class="text-2xl mb-2">🚇</div>
                            <strong>地铁票价</strong>
                            <p class="text-sm text-gray-600 mt-1">不同距离段有不同的单价</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <div class="text-2xl mb-2">💧</div>
                            <strong>阶梯水费</strong>
                            <p class="text-sm text-gray-600 mt-1">用水量越多，单价越高</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <div class="text-2xl mb-2">⚡</div>
                            <strong>阶梯电费</strong>
                            <p class="text-sm text-gray-600 mt-1">用电量分档计价</p>
                        </div>
                    </div>
                </div>

                <div class="bg-indigo-50 rounded-xl p-6">
                    <h4 class="text-lg font-bold text-indigo-800 mb-3">🎯 为什么要用分段函数？</h4>
                    <div class="space-y-3 text-gray-700">
                        <p>• <strong>更符合实际：</strong>很多实际问题的规则本身就是分段的</p>
                        <p>• <strong>政策导向：</strong>通过分段定价引导合理消费（如阶梯电价）</p>
                        <p>• <strong>公平合理：</strong>不同情况采用不同标准（如学生票打折）</p>
                        <p>• <strong>灵活多样：</strong>可以描述更复杂的变化规律</p>
                    </div>
                </div>
            </div>
        `;
    }

    getSlopeConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-purple-100 to-pink-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-purple-800 mb-4">📐 分段函数的变化率</h3>
                    <p class="text-lg text-gray-700 mb-4">
                        在分段函数中，<span class="highlight">不同区间的变化率可能不同</span>，这正是分段函数的特点。
                    </p>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-purple-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">🚗 出租车计费详解</h4>
                    <p class="text-gray-700 mb-3">
                        出租车：起步价10元（2公里内），超出部分 y = 2.7x + 4.6
                    </p>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>公里数(x)</th>
                                <th>车费(y)</th>
                                <th>变化量</th>
                                <th>变化率</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>0-2公里</td>
                                <td>10元</td>
                                <td>0元</td>
                                <td>0元/公里</td>
                            </tr>
                            <tr>
                                <td>2公里</td>
                                <td>10元</td>
                                <td>-</td>
                                <td>分段点</td>
                            </tr>
                            <tr>
                                <td>3公里</td>
                                <td>12.7元</td>
                                <td>+2.7元</td>
                                <td>2.7元/公里</td>
                            </tr>
                            <tr>
                                <td>4公里</td>
                                <td>15.4元</td>
                                <td>+2.7元</td>
                                <td>2.7元/公里</td>
                            </tr>
                            <tr>
                                <td>5公里</td>
                                <td>18.1元</td>
                                <td>+2.7元</td>
                                <td>2.7元/公里</td>
                            </tr>
                        </tbody>
                    </table>
                    <p class="text-gray-700 mt-4">
                        可以看到：<br>
                        • 前2公里：变化率为<strong>0</strong>（固定价格）<br>
                        • 超过2公里：变化率为<strong>2.7</strong>（每公里增加2.7元）
                    </p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-green-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-green-800 mb-3">🚲 共享单车（一次函数）</h4>
                        <p class="text-gray-700 mb-3">
                            匀速10 km/h，每15分钟1.5元<br>
                            函数：y = 0.6x
                        </p>
                        <ul class="space-y-2 text-gray-700">
                            <li>• 变化率恒定：0.6元/公里</li>
                            <li>• 没有分段，是线性关系</li>
                            <li>• 距离越远，费用越高</li>
                            <li>• 适合短中途出行</li>
                        </ul>
                    </div>

                    <div class="bg-blue-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-blue-800 mb-3">🚇 地铁（多段函数）</h4>
                        <p class="text-gray-700 mb-3">
                            分段计价规则：
                        </p>
                        <ul class="space-y-2 text-gray-700 text-sm">
                            <li>• 0-4km: 2元（0.5元/km）</li>
                            <li>• 4-12km: 每4km+1元（0.25元/km）</li>
                            <li>• 12-24km: 每6km+1元（0.167元/km）</li>
                            <li>• >24km: 每8km+1元（0.125元/km）</li>
                        </ul>
                        <p class="text-gray-700 mt-3">
                            <strong>特点：</strong>距离越远，单位价格越低！
                        </p>
                    </div>
                </div>

                <div class="bg-orange-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-orange-800 mb-4">📊 三种交通方式的变化率对比</h4>
                    <div class="bg-white rounded-lg p-4">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>交通方式</th>
                                    <th>函数类型</th>
                                    <th>变化率特点</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>🚕 出租车</td>
                                    <td>分段函数</td>
                                    <td>前2km为0，之后恒定2.7</td>
                                </tr>
                                <tr>
                                    <td>🚲 共享单车</td>
                                    <td>一次函数</td>
                                    <td>恒定0.6</td>
                                </tr>
                                <tr>
                                    <td>🚇 地铁</td>
                                    <td>多段函数</td>
                                    <td>递减（0.5→0.25→0.167→0.125）</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="bg-blue-50 border-l-4 border-blue-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-blue-800 mb-3">💡 理解技巧</h4>
                    <p class="text-gray-700">
                        <strong>分段函数的变化率就像不同路段的限速：</strong><br>
                        • 市区限速40km/h（变化慢）<br>
                        • 高速限速120km/h（变化快）<br>
                        • 不同路段有不同的"速度"（变化率）<br>
                        • 在交界处需要减速或加速（分段点）
                    </p>
                </div>
            </div>
        `;
    }

    getInterceptConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-green-100 to-teal-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-green-800 mb-4">📍 分段函数的初始值</h3>
                    <p class="text-lg text-gray-700 mb-4">
                        在分段函数中，<span class="highlight">初始值</span>（起点值）往往决定了第一段的基准。
                    </p>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-green-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">🚕 出租车的起步价</h4>
                    <p class="text-gray-700 mb-4">
                        出租车计费：y = { 10, (x ≤ 2); 2.7x + 4.6, (x > 2) }
                    </p>
                    <div class="bg-yellow-50 rounded-lg p-4">
                        <p class="text-gray-700">
                            <strong>思考：</strong>为什么叫"起步价"？<br>
                            因为即使你只走0.1公里，也要付<strong>10元</strong>！<br>
                            这个10元就是<span class="highlight">初始值</span>——不管走多远，至少要付这么多。
                        </p>
                    </div>
                    <div class="bg-blue-50 rounded-lg p-4 mt-4">
                        <p class="text-gray-700">
                            <strong>第二段的起点：</strong><br>
                            当x=2时，y=10元<br>
                            当x=3时，y=2.7×3+4.6=12.7元<br>
                            <strong>注意：</strong>第二段函数在x=2处的值也是10元，保证了连续性！
                        </p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="bg-blue-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-blue-800 mb-3">🚲 共享单车</h4>
                        <p class="text-gray-700 mb-3">y = 0.6x</p>
                        <div class="bg-white p-4 rounded-lg">
                            <p class="text-gray-700">
                                <strong>初始值：</strong>0元<br>
                                不骑不花钱，从原点出发<br>
                                <span class="text-green-700 font-bold">没有起步价！</span>
                            </p>
                        </div>
                    </div>

                    <div class="bg-purple-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-purple-800 mb-3">🚇 地铁</h4>
                        <p class="text-gray-700 mb-3 text-sm">0-4km: 2元</p>
                        <div class="bg-white p-4 rounded-lg">
                            <p class="text-gray-700">
                                <strong>初始值：</strong>2元<br>
                                进站就要付2元<br>
                                <span class="text-purple-700 font-bold">最低消费2元</span>
                            </p>
                        </div>
                    </div>

                    <div class="bg-pink-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-pink-800 mb-3">🎓 学生地铁</h4>
                        <p class="text-gray-700 mb-3 text-sm">5折优惠</p>
                        <div class="bg-white p-4 rounded-lg">
                            <p class="text-gray-700">
                                <strong>初始值：</strong>1元<br>
                                学生优惠后<br>
                                <span class="text-pink-700 font-bold">最低1元起</span>
                            </p>
                        </div>
                    </div>
                </div>

                <div class="bg-indigo-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-indigo-800 mb-4">🎯 分段点的连续性</h4>
                    <div class="bg-white rounded-lg p-4">
                        <p class="text-gray-700 mb-3">
                            <strong>什么是连续？</strong><br>
                            在分段点处，左边的函数值 = 右边的函数值
                        </p>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                            <div class="bg-green-50 rounded-lg p-3">
                                <strong class="text-green-700">✓ 连续的例子（出租车）</strong>
                                <p class="text-sm text-gray-700 mt-2">
                                    x=2时：<br>
                                    第一段：y=10<br>
                                    第二段：y=2.7×2+4.6=10<br>
                                    <strong>相等！连续！</strong>
                                </p>
                            </div>
                            <div class="bg-red-50 rounded-lg p-3">
                                <strong class="text-red-700">✗ 不连续的例子</strong>
                                <p class="text-sm text-gray-700 mt-2">
                                    如果规则改为：<br>
                                    x≤2: y=10<br>
                                    x>2: y=3x<br>
                                    x=2时会"跳跃"（10→6）
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">💡 记忆技巧</h4>
                    <p class="text-gray-700">
                        <strong>分段函数的初始值就像入场券：</strong><br>
                        • 有的地方免费进入（共享单车，初始值0）<br>
                        • 有的地方要买门票（出租车起步价10元，地铁2元）<br>
                        • 学生可以打折（地铁学生票1元）<br>
                        • 进去之后，不同区域有不同的收费标准（分段计费）
                    </p>
                </div>
            </div>
        `;
    }

    getGraphConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-orange-100 to-red-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-orange-800 mb-4">📈 分段函数的图像</h3>
                    <p class="text-lg text-gray-700 mb-4">
                        分段函数的图像由<span class="highlight">多段不同的曲线</span>组成，在分段点处可能有"转折"。
                    </p>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-orange-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">✏️ 如何画分段函数图像？</h4>
                    <div class="space-y-4">
                        <div class="solution-step">
                            <span class="solution-step-number">1</span>
                            <div class="solution-step-content">
                                <strong>确定分段点</strong><br>
                                找出函数规则发生变化的x值（如出租车的2公里）
                            </div>
                        </div>
                        <div class="solution-step">
                            <span class="solution-step-number">2</span>
                            <div class="solution-step-content">
                                <strong>分段画图</strong><br>
                                在每个区间内，按照对应的函数关系画图
                            </div>
                        </div>
                        <div class="solution-step">
                            <span class="solution-step-number">3</span>
                            <div class="solution-step-content">
                                <strong>检查连续性</strong><br>
                                在分段点处，检查左右两段是否连接
                            </div>
                        </div>
                        <div class="solution-step">
                            <span class="solution-step-number">4</span>
                            <div class="solution-step-content">
                                <strong>标注关键点</strong><br>
                                标出分段点、起点、特殊点的坐标
                            </div>
                        </div>
                    </div>
                </div>

                <div class="bg-blue-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-blue-800 mb-4">🎯 实战演练：画出出租车计费函数</h4>
                    <div class="bg-white rounded-lg p-4 mb-4">
                        <strong>函数表达式：</strong>
                        <div class="math-formula mt-2">
                            y = { 10, (0 ≤ x ≤ 2)<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.7x + 4.6, (x > 2) }
                        </div>
                    </div>
                    <div class="bg-white rounded-lg p-4 mb-4">
                        <strong>步骤1：列表</strong>
                        <table class="data-table mt-3">
                            <thead>
                                <tr>
                                    <th>x</th>
                                    <th>0</th>
                                    <th>1</th>
                                    <th>2</th>
                                    <th>3</th>
                                    <th>4</th>
                                    <th>5</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>y</strong></td>
                                    <td>10</td>
                                    <td>10</td>
                                    <td>10</td>
                                    <td>12.7</td>
                                    <td>15.4</td>
                                    <td>18.1</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="bg-white rounded-lg p-4">
                        <strong>步骤2：画图</strong>
                        <p class="text-sm text-gray-600 mt-2">
                            • 0-2公里：画一条水平线段（y=10）<br>
                            • 2公里以后：画一条斜向上的直线（斜率2.7）<br>
                            • 在x=2处，两段相连
                        </p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="bg-green-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-green-800 mb-3">🚕 出租车图像</h4>
                        <div class="bg-white p-4 rounded-lg">
                            <p class="text-sm text-gray-700">
                                <strong>特点：</strong><br>
                                • 前段水平（恒定10元）<br>
                                • 后段斜向上（线性增长）<br>
                                • 在x=2处有"拐点"
                            </p>
                        </div>
                    </div>

                    <div class="bg-blue-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-blue-800 mb-3">🚲 单车图像</h4>
                        <div class="bg-white p-4 rounded-lg">
                            <p class="text-sm text-gray-700">
                                <strong>特点：</strong><br>
                                • 过原点的直线<br>
                                • 斜率0.6<br>
                                • 没有分段，一直线性
                            </p>
                        </div>
                    </div>

                    <div class="bg-purple-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-purple-800 mb-3">🚇 地铁图像</h4>
                        <div class="bg-white p-4 rounded-lg">
                            <p class="text-sm text-gray-700">
                                <strong>特点：</strong><br>
                                • 多个分段点<br>
                                • 阶梯状上升<br>
                                • 斜率逐渐变小
                            </p>
                        </div>
                    </div>
                </div>

                <div class="bg-indigo-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-indigo-800 mb-4">📊 三种交通方式图像对比</h4>
                    <div class="bg-white rounded-lg p-4">
                        <p class="text-gray-700 mb-3">
                            在同一坐标系中画出三种交通方式的费用曲线：
                        </p>
                        <ul class="space-y-2 text-gray-700">
                            <li>• <strong class="text-green-700">出租车：</strong>先平后斜，短途贵</li>
                            <li>• <strong class="text-blue-700">共享单车：</strong>直线，最便宜</li>
                            <li>• <strong class="text-purple-700">地铁：</strong>阶梯状，长途划算</li>
                        </ul>
                        <p class="text-gray-700 mt-3">
                            <strong>交点的意义：</strong>两条曲线的交点表示两种方案费用相同的距离
                        </p>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">💡 画图技巧</h4>
                    <p class="text-gray-700">
                        <strong>分段函数画图口诀：</strong><br>
                        分段点处要标清，左右两段仔细分<br>
                        水平线段表恒定，斜线表示在变化<br>
                        阶梯图像多段连，交点位置是关键
                    </p>
                </div>
            </div>
        `;
    }

    getPropertiesConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-pink-100 to-purple-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-pink-800 mb-4">🔍 分段函数的性质</h3>
                    <p class="text-lg text-gray-700">
                        分段函数的性质需要<span class="highlight">分段讨论</span>，不同区间可能有不同的特征。
                    </p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6">
                        <h4 class="text-xl font-bold text-green-800 mb-4">📈 单调性</h4>
                        <div class="space-y-4">
                            <div class="bg-white rounded-lg p-4">
                                <strong class="text-green-700">出租车函数：</strong>
                                <p class="text-gray-700 mt-2">
                                    • 0≤x≤2：<strong>不增不减</strong>（恒为10）<br>
                                    • x>2：<strong>单调递增</strong>（斜率2.7>0）<br>
                                    • 整体：<strong>非递减</strong>（不会下降）
                                </p>
                            </div>
                            <div class="bg-white rounded-lg p-4">
                                <strong class="text-blue-700">共享单车：</strong>
                                <p class="text-gray-700 mt-2">
                                    • 全程：<strong>单调递增</strong><br>
                                    • 变化率恒定为0.6
                                </p>
                            </div>
                            <div class="bg-white rounded-lg p-4">
                                <strong class="text-purple-700">地铁函数：</strong>
                                <p class="text-gray-700 mt-2">
                                    • 全程：<strong>单调递增</strong><br>
                                    • 但增长速度逐渐变慢
                                </p>
                            </div>
                        </div>
                    </div>

                    <div class="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6">
                        <h4 class="text-xl font-bold text-blue-800 mb-4">📍 值域特点</h4>
                        <div class="bg-white rounded-lg p-4 mb-3">
                            <strong class="text-blue-700">出租车（x≥0）：</strong>
                            <p class="text-gray-700 mt-2">
                                • 最小值：10元（起步价）<br>
                                • 值域：[10, +∞)<br>
                                • 不会低于10元
                            </p>
                        </div>
                        <div class="bg-white rounded-lg p-4 mb-3">
                            <strong class="text-green-700">共享单车（x≥0）：</strong>
                            <p class="text-gray-700 mt-2">
                                • 最小值：0元<br>
                                • 值域：[0, +∞)<br>
                                • 从0开始计费
                            </p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-purple-700">地铁（x≥0）：</strong>
                            <p class="text-gray-700 mt-2">
                                • 最小值：2元<br>
                                • 值域：[2, +∞)<br>
                                • 学生票：[1, +∞)
                            </p>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-purple-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">🎨 连续性分析</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-green-50 rounded-lg p-4">
                            <strong class="text-green-700">✓ 连续函数（出租车）</strong>
                            <p class="text-gray-700 mt-2">
                                在x=2处：<br>
                                左极限 = 10<br>
                                右极限 = 2.7×2+4.6 = 10<br>
                                函数值 = 10<br>
                                <strong>三者相等，连续！</strong>
                            </p>
                        </div>
                        <div class="bg-blue-50 rounded-lg p-4">
                            <strong class="text-blue-700">意义</strong>
                            <p class="text-gray-700 mt-2">
                                连续意味着：<br>
                                • 没有"跳跃"<br>
                                • 计费合理<br>
                                • 图像可以一笔画成<br>
                                • 符合实际情况
                            </p>
                        </div>
                    </div>
                </div>

                <div class="bg-orange-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-orange-800 mb-4">⚖️ 费用增长速度对比</h4>
                    <div class="bg-white rounded-lg p-4">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>距离段</th>
                                    <th>🚕出租车</th>
                                    <th>🚲单车</th>
                                    <th>🚇地铁</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>0-2km</td>
                                    <td class="bg-red-50">快（起步价高）</td>
                                    <td class="bg-green-50">慢（0.6/km）</td>
                                    <td class="bg-yellow-50">中（2元固定）</td>
                                </tr>
                                <tr>
                                    <td>2-10km</td>
                                    <td class="bg-yellow-50">中（2.7/km）</td>
                                    <td class="bg-green-50">慢（0.6/km）</td>
                                    <td class="bg-green-50">慢（递减）</td>
                                </tr>
                                <tr>
                                    <td>>10km</td>
                                    <td class="bg-red-50">快（2.7/km）</td>
                                    <td class="bg-green-50">慢（0.6/km）</td>
                                    <td class="bg-green-50">很慢（递减）</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="bg-indigo-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-indigo-800 mb-4">🎯 最优选择分析</h4>
                    <div class="space-y-3 text-gray-700">
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-green-700">短途（0-5km）：</strong>
                            <p class="mt-2">共享单车最便宜，但需要体力</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-blue-700">中途（5-15km）：</strong>
                            <p class="mt-2">地铁和单车都不错，看具体情况</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-purple-700">长途（>15km）：</strong>
                            <p class="mt-2">地铁最划算，尤其是学生票</p>
                        </div>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">💡 性质总结</h4>
                    <p class="text-gray-700">
                        <strong>分段函数性质分析要点：</strong><br>
                        1. 分段讨论：每段单独分析<br>
                        2. 整体把握：综合各段特点<br>
                        3. 关注分段点：连续性、极值<br>
                        4. 实际意义：结合具体问题理解
                    </p>
                </div>
            </div>
        `;
    }

    getApplicationConcept() {
        return `
            <div class="space-y-6">
                <div class="bg-gradient-to-r from-yellow-100 to-orange-100 rounded-xl p-6">
                    <h3 class="text-2xl font-bold text-orange-800 mb-4">🚀 分段函数的实际应用</h3>
                    <p class="text-lg text-gray-700">
                        分段函数在生活中随处可见，是解决实际问题的重要工具！
                    </p>
                </div>

                <div class="bg-white rounded-xl p-6 border-2 border-orange-200">
                    <h4 class="text-xl font-bold text-gray-800 mb-4">🚕 经典案例：出行方案决策</h4>
                    <div class="bg-blue-50 rounded-lg p-4 mb-4">
                        <strong>问题情境：</strong>
                        <p class="text-gray-700 mt-2">
                            小明要去15公里外的地方，有三种出行方式：<br>
                            <strong>🚕 出租车：</strong>起步价10元（2km内），超出部分 y = 2.7x + 4.6<br>
                            <strong>🚲 共享单车：</strong>y = 0.6x（匀速10km/h）<br>
                            <strong>🚇 地铁：</strong>分段计价，学生5折<br>
                            <strong>问题：</strong>如何选择最省钱？
                        </p>
                    </div>

                    <div class="space-y-4">
                        <div class="solution-step">
                            <span class="solution-step-number">1</span>
                            <div class="solution-step-content">
                                <strong>建立函数模型</strong><br>
                                出租车：y₁ = { 10, (x≤2); 2.7x+4.6, (x>2) }<br>
                                共享单车：y₂ = 0.6x<br>
                                地铁：y₃ = 分段计价函数<br>
                                学生地铁：y₄ = 0.5 × y₃
                            </div>
                        </div>

                        <div class="solution-step">
                            <span class="solution-step-number">2</span>
                            <div class="solution-step-content">
                                <strong>计算15公里的费用</strong><br>
                                出租车：y₁ = 2.7×15 + 4.6 = 45.1元<br>
                                共享单车：y₂ = 0.6×15 = 9元<br>
                                地铁：y₃ = 5元（12-24km段）<br>
                                学生地铁：y₄ = 2.5元
                            </div>
                        </div>

                        <div class="solution-step">
                            <span class="solution-step-number">3</span>
                            <div class="solution-step-content">
                                <strong>综合分析</strong><br>
                                • 最便宜：学生地铁（2.5元）<br>
                                • 次便宜：普通地铁（5元）<br>
                                • 第三：共享单车（9元，但需骑90分钟）<br>
                                • 最贵：出租车（45.1元，但最快最舒适）
                            </div>
                        </div>

                        <div class="solution-step">
                            <span class="solution-step-number">4</span>
                            <div class="solution-step-content">
                                <strong>决策建议</strong><br>
                                • 学生优先选地铁（省钱）<br>
                                • 赶时间选出租车<br>
                                • 锻炼身体选单车<br>
                                • 综合考虑时间、费用、舒适度
                            </div>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-green-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-green-800 mb-3">💧 案例2：阶梯水费</h4>
                        <p class="text-gray-700 mb-3">
                            <strong>计费规则：</strong><br>
                            0-10吨：2元/吨<br>
                            10-20吨：3元/吨<br>
                            >20吨：5元/吨
                        </p>
                        <div class="bg-white rounded-lg p-3">
                            <strong>分段函数：</strong><br>
                            y = { 2x, (0≤x≤10)<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;20+3(x-10), (10<x≤20)<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;50+5(x-20), (x>20) }
                        </div>
                    </div>

                    <div class="bg-purple-50 rounded-xl p-6">
                        <h4 class="text-lg font-bold text-purple-800 mb-3">⚡ 案例3：阶梯电费</h4>
                        <p class="text-gray-700 mb-3">
                            <strong>计费规则：</strong><br>
                            0-200度：0.5元/度<br>
                            200-400度：0.6元/度<br>
                            >400度：0.8元/度
                        </p>
                        <div class="bg-white rounded-lg p-3">
                            <strong>目的：</strong><br>
                            鼓励节约用电，<br>
                            用得越多，单价越高
                        </div>
                    </div>
                </div>

                <div class="bg-indigo-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-indigo-800 mb-4">🎯 解决分段函数问题的步骤</h4>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div class="bg-white rounded-lg p-4 text-center">
                            <div class="text-3xl mb-2">1️⃣</div>
                            <strong>识别分段</strong>
                            <p class="text-sm text-gray-600 mt-1">找出分段点和各段规则</p>
                        </div>
                        <div class="bg-white rounded-lg p-4 text-center">
                            <div class="text-3xl mb-2">2️⃣</div>
                            <strong>建立模型</strong>
                            <p class="text-sm text-gray-600 mt-1">写出分段函数表达式</p>
                        </div>
                        <div class="bg-white rounded-lg p-4 text-center">
                            <div class="text-3xl mb-2">3️⃣</div>
                            <strong>分段计算</strong>
                            <p class="text-sm text-gray-600 mt-1">根据x的范围选择公式</p>
                        </div>
                        <div class="bg-white rounded-lg p-4 text-center">
                            <div class="text-3xl mb-2">4️⃣</div>
                            <strong>综合分析</strong>
                            <p class="text-sm text-gray-600 mt-1">比较不同方案优劣</p>
                        </div>
                    </div>
                </div>

                <div class="bg-pink-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-pink-800 mb-4">🎓 数学建模思想</h4>
                    <div class="space-y-3 text-gray-700">
                        <div class="bg-white rounded-lg p-4">
                            <strong>1. 抽象化：</strong>
                            <p class="mt-1">把实际问题转化为数学模型</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong>2. 数学化：</strong>
                            <p class="mt-1">用函数表达式描述规律</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong>3. 求解：</strong>
                            <p class="mt-1">运用数学方法计算和分析</p>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong>4. 还原：</strong>
                            <p class="mt-1">把数学结果转化为实际建议</p>
                        </div>
                    </div>
                </div>

                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg">
                    <h4 class="text-lg font-bold text-yellow-800 mb-3">💡 应用技巧</h4>
                    <ul class="space-y-2 text-gray-700">
                        <li><strong>1. 识别关键词：</strong>"起步价"、"超出部分"、"分段计价"等</li>
                        <li><strong>2. 画图辅助：</strong>用图像直观比较不同方案</li>
                        <li><strong>3. 找临界点：</strong>分段点往往是决策的关键</li>
                        <li><strong>4. 验证连续性：</strong>确保在分段点处计费合理</li>
                        <li><strong>5. 综合考虑：</strong>不只看价格，还要考虑时间、舒适度等</li>
                    </ul>
                </div>

                <div class="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6">
                    <h4 class="text-xl font-bold text-indigo-800 mb-4">🌟 学习收获</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-blue-700">数学能力：</strong>
                            <ul class="mt-2 space-y-1 text-sm text-gray-700">
                                <li>✓ 理解分段函数概念</li>
                                <li>✓ 掌握分段建模方法</li>
                                <li>✓ 学会分段计算技巧</li>
                                <li>✓ 培养数学建模思想</li>
                            </ul>
                        </div>
                        <div class="bg-white rounded-lg p-4">
                            <strong class="text-purple-700">实践能力：</strong>
                            <ul class="mt-2 space-y-1 text-sm text-gray-700">
                                <li>✓ 解决实际出行问题</li>
                                <li>✓ 做出理性经济决策</li>
                                <li>✓ 理解社会计费规则</li>
                                <li>✓ 培养节约意识</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    generateCustomExplanation(question) {
        // 根据问题关键词生成相应讲解
        const keywords = {
            '分段': 'basic',
            '变化率': 'slope',
            '初始值': 'intercept',
            '起步价': 'intercept',
            '图像': 'graph',
            '性质': 'properties',
            '应用': 'application',
            '画': 'graph',
            '单调': 'properties',
            '连续': 'properties',
            '出租车': 'application',
            '地铁': 'application',
            '单车': 'application'
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
                        分段函数是指在不同的自变量范围内，有不同对应关系的函数。理解它的关键是：
                    </p>
                    <ul class="mt-3 space-y-2 text-gray-700">
                        <li>• <strong>分段点</strong>是函数规则变化的临界值</li>
                        <li>• <strong>每段</strong>有各自的函数表达式</li>
                        <li>• <strong>连续性</strong>在分段点处很重要</li>
                        <li>• 在生活中有广泛应用，如出租车计费、阶梯电价等</li>
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
