import streamlit as st
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
from deploy_utils import get_nltk_data_dir
from ui_theme import inject_iekg_theme, render_guide_card, render_hero

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保 nltk 资源已下载
@st.cache_resource
def download_nltk_resources():
    nltk_data_dir = get_nltk_data_dir()
    nltk_data_dir.mkdir(parents=True, exist_ok=True)

    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.append(str(nltk_data_dir))

    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab/english',
        'stopwords': 'corpora/stopwords',
    }

    for package_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, download_dir=str(nltk_data_dir), quiet=True)

    return True

download_nltk_resources()

# 设置页面配置
st.set_page_config(
    page_title='NLP 教学 Web 应用',
    page_icon='📚',
    layout='wide',
    initial_sidebar_state='auto'
)

inject_iekg_theme(
    """
    .plot-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,251,255,0.92));
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    """
)

render_hero(
    "语义分析综合测试平台",
    "Semantic Analysis Playground",
    "围绕传统统计语义、词向量训练、预训练嵌入和句子级表示，提供一个适合课堂实验和交互测试的综合语义分析工作台。",
    [
        ("Step 01", "TF-IDF + LSA"),
        ("Step 02", "Word2Vec"),
        ("Step 03", "GloVe"),
        ("Step 04", "FastText"),
    ],
)

# 文本输入框
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-heading">
        <div>
            <div class="section-kicker">Workspace</div>
            <h3>文本语料输入</h3>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_guide_card(
    "这个输入区是后面多个模块共享的实验语料入口，适合粘贴一段几百词左右的英文文本做统一分析。",
    "输入完成后，各个标签页会围绕同一份文本去做统计语义、词向量训练和句子级表示计算。",
    "如果只想看 GloVe 预训练演示，可以不填文本；其余模块建议输入一段完整英文段落再体验。",
)
default_corpus = """
Natural language processing is changing the way people interact with computers. In a modern classroom, students can explore how words carry meaning, how context changes interpretation, and how statistical patterns reveal hidden structure in text. A short news article may describe a city government meeting, while a scientific report may explain how researchers analyze language data with machine learning tools.

Semantic analysis helps us move beyond counting words. If two sentences use different expressions to describe the same event, a good semantic model should still recognize their similarity. For example, the sentence "The committee approved the plan" is close in meaning to "The proposal was accepted by the committee," even though the surface forms are different. This difference between form and meaning is an important topic in computational linguistics.

Traditional approaches such as TF-IDF and latent semantic analysis focus on distributional statistics. They measure how often words appear in documents and how frequently terms occur together. These methods are simple, interpretable, and useful for keyword extraction, topic exploration, and document comparison. However, they may struggle when synonyms appear in different contexts or when rare words carry important meaning.

Neural word embeddings offer another perspective. Word2Vec learns vector representations by predicting surrounding words, while FastText improves robustness by modeling subword information. Because FastText uses character n-grams, it can often produce reasonable vectors for unseen or misspelled words such as "computeer" or "langauge". In teaching demonstrations, this contrast helps students understand why subword modeling matters.

Sentence representations are also important. When we average word vectors, we can build a simple sentence embedding and estimate similarity between two statements. Although this method is not as powerful as large transformer models, it is intuitive and easy to explain. Students can quickly observe that sentences about education, research, and language technology tend to cluster together in semantic space.

As AI systems become more common, semantic analysis is no longer only a research topic. It supports search engines, recommendation systems, chatbots, machine translation, and document understanding. By experimenting with the same corpus across multiple modules, learners can compare classic statistical models with modern embedding-based approaches and build a clearer intuition about how machines represent meaning.
""".strip()

corpus_input = st.text_area(
    '请输入英文文本语料（约 500-1000 词）：',
    value=default_corpus,
    height=300,
)

# 提示信息
if not corpus_input:
    st.info('请在上方输入英文文本语料，以便使用各个模块的功能。')
else:
    st.success(f'已输入 {len(corpus_input)} 个字符的文本语料。')
st.markdown("</div>", unsafe_allow_html=True)

# 四个标签页
tab1, tab2, tab3, tab4 = st.tabs(['传统统计模型', 'Word2Vec 实时训练', 'GloVe 预训练模型', 'FastText 与句子级表示'])

# 模块 1：传统统计模型
with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 01</div>
                <h3>传统统计模型</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("从分句、TF-IDF 到 LSA 和共现分析，展示经典统计语义方法如何理解文本。")
    render_guide_card(
        "这一部分演示不依赖深度学习时，如何用词频、句子矩阵和降维方法提取文本里的显著语义结构。",
        "先看 TF-IDF 和关键词，再看同现词对与二维可视化，能快速建立对文本主题的整体感知。",
        "如果输入文本较短，矩阵和共现结构会相对稀疏，这是正常现象。",
    )
    
    # 添加模块说明
    with st.expander('模块说明', expanded=False):
        st.write('''
        **功能**：使用传统统计方法分析文本，包括分句、TF-IDF 计算、关键词提取和词汇可视化。
        
        **原理**：
        - **TF-IDF**：词频-逆文档频率，用于评估词语对文档集合的重要程度。
        - **TruncatedSVD**：截断奇异值分解，用于降维，将高维词汇空间映射到 2 维平面。
        
        **使用方法**：在顶部输入英文文本，系统会自动分词、计算 TF-IDF 矩阵、提取关键词并生成可视化散点图。
        ''')
    
    if corpus_input:
        # 添加文本清洗选项
        enable_cleaning = st.checkbox('启用文本清洗（过滤数字/停用词/合并连字符）', value=False)
        
        # 分句
        sentences = nltk.sent_tokenize(corpus_input)
        st.write(f'输入文本被分割为 {len(sentences)} 个句子')
        
        # 文本清洗功能
        if enable_cleaning:
            import re
            from nltk.corpus import stopwords
            
            cleaned_sentences = []
            stop_words = set(stopwords.words('english'))
            
            for sentence in sentences:
                # 合并连字符
                sentence = re.sub(r'(\w)-(\w)', r'\1\2', sentence)
                # 过滤数字
                sentence = re.sub(r'\d+', '', sentence)
                # 分词并过滤停用词
                words = nltk.word_tokenize(sentence.lower())
                filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
                cleaned_sentence = ' '.join(filtered_words)
                if cleaned_sentence:
                    cleaned_sentences.append(cleaned_sentence)
            
            sentences = cleaned_sentences
            st.write(f'清洗后剩余 {len(sentences)} 个句子')
        
        # 计算 TF-IDF 矩阵
        vectorizer = TfidfVectorizer(stop_words='english' if not enable_cleaning else None)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # 展示 TF-IDF 矩阵（前 10 行和前 10 列）
        st.subheader('TF-IDF 矩阵')
        feature_names = vectorizer.get_feature_names_out()
        n_rows = min(10, len(sentences))
        n_cols = min(10, len(feature_names))
        
        # 转换为DataFrame并设置列名和索引
        import pandas as pd
        df = pd.DataFrame(
            tfidf_matrix[:n_rows, :n_cols].toarray(),
            columns=feature_names[:n_cols],
            index=[f'句子 {i+1}' for i in range(n_rows)]
        )
        st.dataframe(df)
        st.write(f'完整矩阵形状: {tfidf_matrix.shape} (句子数 × 词汇数)')
        
        # 提取全局权重最高的 5 个关键词
        tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = tfidf_scores.argsort()[-5:][::-1]
        top_keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
        
        st.subheader('全局权重最高的 5 个关键词')
        for keyword, score in top_keywords:
            st.write(f'{keyword}: {score:.4f}')
        
        # 基于共现矩阵的 LSA 分析
        st.subheader('基于共现矩阵的 LSA 分析')
        
        # 计算共现矩阵
        from sklearn.feature_extraction.text import CountVectorizer
        count_vectorizer = CountVectorizer(stop_words='english')
        count_matrix = count_vectorizer.fit_transform(sentences)
        co_occurrence_matrix = count_matrix.T.dot(count_matrix)
        
        # 过滤共现次数<2的单元格
        co_occurrence_matrix[co_occurrence_matrix < 2] = 0
        
        # 使用 SVD 进行 LSA 分解
        lsa = TruncatedSVD(n_components=10, random_state=42)
        lsa.fit(co_occurrence_matrix)
        
        # 使用滑动窗口统计共现次数
        st.subheader('同现频率最高的词对')
        
        # 文本清洗和分词
        import re
        from nltk.corpus import stopwords
        
        # 收集所有词语
        all_words = []
        stop_words = set(stopwords.words('english'))
        
        for sentence in sentences:
            # 分词（保留连字符）
            words = re.findall(r'\b\w+(?:-\w+)*\b', sentence.lower())
            # 过滤
            filtered_words = []
            for word in words:
                # 过滤数字
                if any(char.isdigit() for char in word):
                    continue
                # 过滤停用词
                if word in stop_words:
                    continue
                # 过滤长度≤2的词
                if len(word) <= 2:
                    continue
                filtered_words.append(word)
            all_words.extend(filtered_words)
        
        # 滑动窗口统计共现次数
        window_size = 5
        co_occurrence_counts = {}
        
        for i in range(len(all_words)):
            # 窗口范围
            start = max(0, i - window_size // 2)
            end = min(len(all_words), i + window_size // 2 + 1)
            
            # 窗口内的其他词
            for j in range(start, end):
                if i != j:
                    word1 = all_words[i]
                    word2 = all_words[j]
                    # 确保词对顺序一致（小的在前）
                    if word1 > word2:
                        word1, word2 = word2, word1
                    # 统计共现次数
                    pair = (word1, word2)
                    co_occurrence_counts[pair] = co_occurrence_counts.get(pair, 0) + 1
        
        # 添加最小共现次数滑动条
        min_count = st.slider('最小共现次数', min_value=1, max_value=5, value=2, step=1)
        
        # 筛选并排序
        filtered_pairs = [(word1, word2, count) for (word1, word2), count in co_occurrence_counts.items() if count >= min_count]
        filtered_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = filtered_pairs[:5]
        
        # 输出结果
        if top_pairs:
            for word1, word2, count in top_pairs:
                st.write(f'{word1} — {word2} : 共现 {count} 次')
        else:
            st.write('没有找到符合条件的词对')
        
        # 使用 TruncatedSVD 降维到 2 维（使用过滤后的共现矩阵）
        svd = TruncatedSVD(n_components=2, random_state=42)
        word_vectors = svd.fit_transform(co_occurrence_matrix)
        
        # 获取特征名称
        lsa_feature_names = count_vectorizer.get_feature_names_out()
        
        # 绘制散点图
        st.subheader('词汇 2D 可视化')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(word_vectors[:, 0], word_vectors[:, 1], alpha=0.5)
        
        # 标注部分词汇（共现次数较高的前20个）
        # 计算每个词的总共现次数
        total_co_occurrence = co_occurrence_matrix.sum(axis=1)
        # 转换为NumPy数组
        total_co_occurrence = np.asarray(total_co_occurrence).flatten()
        # 获取总共现次数最高的前20个词的索引
        top_20_indices = total_co_occurrence.argsort()[-20:][::-1]
        for i in top_20_indices:
            if total_co_occurrence[i] > 0:  # 确保只标注有共现的词
                ax.annotate(lsa_feature_names[i], (word_vectors[i, 0], word_vectors[i, 1]))
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title('LSA 词汇降维可视化（过滤共现次数<2）')
        st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info('请在顶部输入文本语料以使用此模块')
    st.markdown("</div>", unsafe_allow_html=True)

# 模块 2：Word2Vec 实时训练
with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 02</div>
                <h3>Word2Vec 实时训练</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("在当前输入语料上现场训练词向量，观察 CBOW 与 Skip-Gram 的差异。")
    render_guide_card(
        "这一部分演示词向量如何从局部上下文里学习语义相似性，也就是“相似词在向量空间里更接近”。",
        "选好训练架构和窗口大小后，输入一个词，就能查看它在当前语料下最接近的若干邻居。",
        "因为是现场小语料训练，结果会非常依赖你输入的文本内容。",
    )
    
    # 添加模块说明
    with st.expander('模块说明', expanded=False):
        st.write('''
        **功能**：使用 Word2Vec 模型实时训练词向量，并查询相似词汇。
        
        **原理**：
        - **CBOW**：连续词袋模型，通过上下文预测中心词。
        - **Skip-Gram**：通过中心词预测上下文。
        - **词向量**：将词语映射到低维向量空间，相似词在空间中距离较近。
        
        **使用方法**：在顶部输入英文文本，选择训练架构和参数，然后输入一个单词查询相似词汇。
        ''')
    
    if corpus_input:
        # 分词
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(corpus_input)]
        
        # 参数设置
        architecture = st.radio('训练架构', ['CBOW (sg=0)', 'Skip-Gram (sg=1)'])
        sg = 1 if architecture == 'Skip-Gram (sg=1)' else 0
        
        window = st.slider('window 大小', 2, 10, 5)
        vector_size = st.slider('vector_size 大小', 50, 300, 100)
        
        # 训练模型
        model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=window, sg=sg, min_count=1)
        
        # 单词输入框
        word_input = st.text_input('输入一个单词，查找最相似的词汇：')
        
        if word_input:
            try:
                similar_words = model.wv.most_similar(word_input, topn=5)
                st.subheader(f'与 "{word_input}" 最相似的 5 个词汇：')
                for word, similarity in similar_words:
                    st.write(f'{word}: {similarity:.4f}')
            except KeyError:
                st.error(f'单词 "{word_input}" 不在词汇表中')
    else:
        st.info('请在顶部输入文本语料以使用此模块')
    st.markdown("</div>", unsafe_allow_html=True)

# 模块 3：GloVe 预训练模型
with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 03</div>
                <h3>GloVe 预训练模型</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("直接调用预训练 GloVe 词向量做类比推理和相似度计算。")
    render_guide_card(
        "这一部分用现成的大规模预训练词向量，演示词语之间的几何关系如何支撑类比计算。",
        "上半部分是 A:B::C:? 形式的词类比，下半部分是两个词之间的余弦相似度。",
        "如果首次加载较慢，通常是因为正在下载预训练模型文件。",
    )
    
    # 添加模块说明
    with st.expander('模块说明', expanded=False):
        st.write('''
        **功能**：使用预训练的 GloVe 模型进行词类比和单词相似度计算。
        
        **原理**：
        - **GloVe**：全局词向量，结合了全局统计信息和局部上下文信息。
        - **词类比**：通过向量运算实现，如 "国王 - 男人 + 女人 = 女王"。
        - **相似度计算**：使用余弦相似度衡量两个词向量之间的相似程度。
        
        **使用方法**：无需输入语料，直接使用预训练模型进行词类比和相似度计算。
        ''')
    
    # 缓存预训练模型
    @st.cache_resource
    def load_glove_model():
        return api.load('glove-wiki-gigaword-100')
    
    try:
        glove_model = load_glove_model()
        
        # 词类比计算器
        st.subheader('词类比计算器')
        col1, col2, col3 = st.columns(3)
        with col1:
            word_a = st.text_input('单词 A')
        with col2:
            word_b = st.text_input('单词 B')
        with col3:
            word_c = st.text_input('单词 C')
        
        if word_a and word_b and word_c:
            try:
                result = glove_model.most_similar(positive=[word_b, word_c], negative=[word_a], topn=1)
                st.write(f'结果: {result[0][0]} (相似度: {result[0][1]:.4f})')
            except KeyError as e:
                st.error(f'错误: {e} 不在模型词汇表中')
        
        # 单词相似度计算
        st.subheader('单词相似度计算')
        col4, col5 = st.columns(2)
        with col4:
            word1 = st.text_input('单词 1')
        with col5:
            word2 = st.text_input('单词 2')
        
        if word1 and word2:
            try:
                similarity = glove_model.similarity(word1, word2)
                st.write(f'"{word1}" 和 "{word2}" 的相似度: {similarity:.4f}')
            except KeyError as e:
                st.error(f'错误: {e} 不在模型词汇表中')
    except Exception as e:
        st.error(f'加载 GloVe 模型时出错: {e}')
    st.markdown("</div>", unsafe_allow_html=True)

# 模块 4：FastText 与句子级表示
with tab4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 04</div>
                <h3>FastText 与句子级表示</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("比较 FastText 对未登录词的处理能力，并用平均词向量做句子相似度。")
    render_guide_card(
        "这一部分突出 FastText 的子词建模优势，以及如何从词向量进一步得到简单的句子级表示。",
        "先看 OOV 测试里 Word2Vec 和 FastText 的不同表现，再看两句文本的余弦相似度。",
        "拼写错误词越接近真实词形，FastText 通常越容易给出合理邻居。",
    )
    
    # 添加模块说明
    with st.expander('模块说明', expanded=False):
        st.write('''
        **功能**：使用 FastText 模型处理未登录词（OOV）和计算句子相似度。
        
        **原理**：
        - **FastText**：Word2Vec 的扩展，通过子词信息处理未登录词。
        - **OOV 处理**：利用字符 n-gram 信息，即使是拼写错误的单词也能得到词向量。
        - **Sent2Vec**：通过词向量的平均计算句子向量，衡量句子间相似度。
        
        **使用方法**：在顶部输入英文文本，测试 FastText 对未登录词的处理能力，以及计算两个句子的相似度。
        ''')
    
    if corpus_input:
        # 分词
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(corpus_input)]
        
        # 训练 FastText 模型
        fasttext_model = FastText(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
        
        # 训练 Word2Vec 模型用于比较
        word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
        
        # OOV 测试
        st.subheader('OOV 测试（拼写错误的单词）')
        oov_word = st.text_input('输入一个拼写错误的单词（如 "computeer"）：')
        
        if oov_word:
            st.write('Word2Vec 结果：')
            try:
                similar_words_w2v = word2vec_model.wv.most_similar(oov_word, topn=3)
                for word, similarity in similar_words_w2v:
                    st.write(f'{word}: {similarity:.4f}')
            except KeyError:
                st.write('未登录词')
            
            st.write('FastText 结果：')
            try:
                similar_words_ft = fasttext_model.wv.most_similar(oov_word, topn=3)
                for word, similarity in similar_words_ft:
                    st.write(f'{word}: {similarity:.4f}')
            except Exception as e:
                st.error(f'错误: {e}')
        
        # Sent2Vec 简单实现
        st.subheader('Sent2Vec 句子相似度')
        sentence1 = st.text_input('句子 1')
        sentence2 = st.text_input('句子 2')
        
        if sentence1 and sentence2:
            # 分词
            tokens1 = nltk.word_tokenize(sentence1)
            tokens2 = nltk.word_tokenize(sentence2)
            
            # 计算句向量
            def get_sentence_vector(tokens, model):
                vectors = []
                for token in tokens:
                    try:
                        vectors.append(model.wv[token])
                    except KeyError:
                        pass
                if vectors:
                    return np.mean(vectors, axis=0)
                else:
                    return np.zeros(100)
            
            vec1 = get_sentence_vector(tokens1, fasttext_model)
            vec2 = get_sentence_vector(tokens2, fasttext_model)
            
            # 计算余弦相似度
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            st.write(f'两个句子的相似度: {similarity:.4f}')
    else:
        st.info('请在顶部输入文本语料以使用此模块')
    st.markdown("</div>", unsafe_allow_html=True)
