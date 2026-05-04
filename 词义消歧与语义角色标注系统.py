import streamlit as st
import nltk
import numpy as np
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from ui_theme import inject_iekg_theme, render_guide_card, render_hero
from deploy_utils import get_nltk_data_dir

# 调试信息
print(f"Python version: {sys.version}")
print(f"Streamlit version: {st.__version__}")
print(f"Current directory: {os.getcwd()}")
print(f"Script path: {os.path.abspath(__file__)}")

# 尝试导入依赖库
HAS_SPACY = False
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    st.warning('未安装 spacy 库，语义角色标注模块将不可用。')
    st.info('可以通过运行: pip install --user spacy 来安装这个库。')

# 尝试导入 transformers 相关库
HAS_TRANSFORMERS = False
TRANSFORMERS_IMPORT_ERROR = ""
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except Exception as exc:
    TRANSFORMERS_IMPORT_ERROR = str(exc)
    st.warning('transformers / torch 依赖未正确加载，词义消歧模块的 BERT 功能将不可用。')
    st.info('如果这是部署环境，请确认 requirements.txt 中包含 transformers 和 torch，并查看 Cloud logs。')
    st.caption(f'具体导入错误：{TRANSFORMERS_IMPORT_ERROR}')

def download_nltk_resources():
    try:
        # 使用跨平台可写目录，避免部署到 Linux 云环境时路径失效
        nltk_data_dir = get_nltk_data_dir()
        nltk_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加自定义数据目录到 NLTK 路径
        if str(nltk_data_dir) not in nltk.data.path:
            nltk.data.path.append(str(nltk_data_dir))
        
        # 仅在资源缺失时下载，减少云端冷启动时间
        resources = {
            'punkt': 'tokenizers/punkt',
            'punkt_tab': 'tokenizers/punkt_tab/english',
            'wordnet': 'corpora/wordnet',
            'omw-1.4': 'corpora/omw-1.4',
        }
        for package_name, resource_path in resources.items():
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(package_name, download_dir=str(nltk_data_dir), quiet=True)
        
        return True
    except Exception as e:
        st.error(f'下载 NLTK 资源时出错: {e}')
        st.info('部署环境一般不需要管理员权限，请优先检查缓存目录是否可写以及网络是否正常。')
        st.info('如果本地手动处理，可以运行: python -m nltk.downloader punkt punkt_tab wordnet omw-1.4')
        return False

download_nltk_resources()

# 加载 spaCy 模型
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except Exception as e:
        st.error(f'加载 spaCy 模型失败: {e}')
        st.info('正在尝试下载 spaCy 模型...')
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                st.success('spaCy 模型下载成功')
                return spacy.load('en_core_web_sm')
            else:
                st.error(f'下载 spaCy 模型失败: {result.stderr}')
                st.stop()
        except Exception as e2:
            st.error(f'下载 spaCy 模型时出错: {e2}')
            st.stop()

# 加载 BERT 模型和分词器
@st.cache_resource
def load_bert_model():
    if not HAS_TRANSFORMERS:
        return None, None
    try:
        model = AutoModel.from_pretrained('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return model, tokenizer
    except Exception as e:
        st.error(f'加载 BERT 模型失败: {e}')
        st.info('首次运行时需要下载 BERT 模型，这可能需要一些时间...')
        return None, None

# 设置页面配置
st.set_page_config(
    page_title='深层语义分析平台',
    page_icon='📚',
    layout='wide',
    initial_sidebar_state='auto'
)

inject_iekg_theme(
    """
    .stCard, .result-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,251,255,0.92));
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .stExpander {
        border-radius: 18px !important;
    }
    """
)

render_hero(
    "深层语义分析平台",
    "Deep Semantic Analysis Studio",
    "统一采用信息抽取页面的视觉风格，把词义消歧和语义角色标注放进更适合课堂展示的双模块语义分析工作台。",
    [
        ("Step 01", "Lesk Baseline"),
        ("Step 02", "BERT Context"),
        ("Step 03", "SRL Heuristics"),
    ],
)

# 两个标签页
tab1, tab2 = st.tabs(['🔤 词义消歧（WSD）对比测试', '🎭 语义角色标注（SRL）近似提取与可视化'])

# 模块一：词义消歧（WSD）对比测试
with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 01</div>
                <h3>词义消歧（WSD）对比测试</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("对比传统 Lesk 方法和上下文向量方法，让多义词的语境差异一眼可见。")
    render_guide_card(
        "这一部分围绕同一个多义词，比较不同语境下系统会如何判断它到底是哪一种词义。",
        "先看 Lesk 的定义匹配结果，再看 BERT 上下文表示给出的相似度和预测词义。",
        "最适合演示的词通常是 bank、bat、plant 这种上下文差异很明显的英文多义词。",
    )
    
    # 添加模块说明
    with st.expander('📝 模块说明', expanded=True):
        st.markdown('''
        **功能**：对比传统 Lesk 算法和基于 BERT 上下文向量的词义消歧方法。
        
        **原理**：
        - **Lesk 算法**：基于词典定义的重叠度进行词义消歧。
        - **BERT 上下文向量**：使用预训练语言模型提取词的上下文表示，通过余弦相似度衡量不同语境下的差异。
        
        **使用方法**：输入包含同一多义词的两个句子，系统会分别使用两种方法进行分析。
        ''')
    
    # 文本输入框
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        sentence1 = st.text_area('句子 1（包含多义词）：', value='I went to the bank to deposit my money.', height=100)
    
    with col2:
        sentence2 = st.text_area('句子 2（包含多义词）：', value='I sat by the river bank.', height=100)
    
    target_word = st.text_input('目标多义词：', value='bank')
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 分析按钮
    if st.button('🚀 开始分析'):
        # 传统方法：Lesk 算法
        st.subheader('📚 传统方法：Lesk 算法')
        
        from nltk.wsd import lesk
        from nltk.tokenize import word_tokenize
        
        # 分析句子 1
        tokens1 = word_tokenize(sentence1)
        synset1 = lesk(tokens1, target_word)
        
        # 分析句子 2
        tokens2 = word_tokenize(sentence2)
        synset2 = lesk(tokens2, target_word)
        
        # 显示结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.write(f'**句子 1**: {sentence1}')
            if synset1:
                st.write(f'**预测词义**: {synset1.name()}')
                st.write(f'**定义**: {synset1.definition()}')
            else:
                st.warning('Lesk 算法未找到合适的词义定义')
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.write(f'**句子 2**: {sentence2}')
            if synset2:
                st.write(f'**预测词义**: {synset2.name()}')
                st.write(f'**定义**: {synset2.definition()}')
            else:
                st.warning('Lesk 算法未找到合适的词义定义')
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 基于上下文向量的方法
        st.subheader('🤖 基于上下文向量的方法')
        
        if HAS_TRANSFORMERS:
            try:
                model, tokenizer = load_bert_model()
                
                if model is not None and tokenizer is not None:
                    # 提取句子 1 中目标词的向量
                    inputs1 = tokenizer(sentence1, return_tensors='pt')
                    with torch.no_grad():
                        outputs1 = model(**inputs1)
                    
                    # 找到目标词的位置
                    tokens1 = tokenizer.tokenize(sentence1)
                    target_positions1 = []
                    for i, token in enumerate(tokens1):
                        if target_word.lower() in token.lower():
                            target_positions1.append(i)
                    
                    if target_positions1:
                        # 取目标词所有子词的向量平均
                        word_embedding1 = outputs1.last_hidden_state[0, target_positions1, :].mean(dim=0).numpy()
                    else:
                        st.error(f'目标词 "{target_word}" 不在句子 1 中')
                        word_embedding1 = None
                    
                    # 提取句子 2 中目标词的向量
                    inputs2 = tokenizer(sentence2, return_tensors='pt')
                    with torch.no_grad():
                        outputs2 = model(**inputs2)
                    
                    # 找到目标词的位置
                    tokens2 = tokenizer.tokenize(sentence2)
                    target_positions2 = []
                    for i, token in enumerate(tokens2):
                        if target_word.lower() in token.lower():
                            target_positions2.append(i)
                    
                    if target_positions2:
                        # 取目标词所有子词的向量平均
                        word_embedding2 = outputs2.last_hidden_state[0, target_positions2, :].mean(dim=0).numpy()
                    else:
                        st.error(f'目标词 "{target_word}" 不在句子 2 中')
                        word_embedding2 = None
                    
                    # 计算余弦相似度
                    if word_embedding1 is not None and word_embedding2 is not None:
                        similarity = cosine_similarity([word_embedding1], [word_embedding2])[0][0]
                        
                        # 显示相似度结果
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.write(f'**BERT 上下文向量相似度**: {similarity:.4f}')
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 基于上下文向量方法预测词义
                        def get_synset_embedding(synset, tokenizer, model):
                            # 使用词义定义作为输入获取向量
                            definition = synset.definition()
                            inputs = tokenizer(definition, return_tensors='pt', truncation=True, max_length=512)
                            with torch.no_grad():
                                outputs = model(**inputs)
                            # 使用[CLS] token的向量作为词义表示
                            return outputs.last_hidden_state[0, 0, :].numpy()
                        
                        # 预测句子1中的词义
                        from nltk.corpus import wordnet as wn
                        synsets = wn.synsets(target_word)
                        
                        if synsets:
                            # 计算每个词义与目标词上下文向量的相似度
                            synset_similarities1 = []
                            for synset in synsets:
                                try:
                                    synset_emb = get_synset_embedding(synset, tokenizer, model)
                                    sim = cosine_similarity([word_embedding1], [synset_emb])[0][0]
                                    synset_similarities1.append((synset, sim))
                                except Exception as e:
                                    continue
                            
                            # 选择相似度最高的词义
                            if synset_similarities1:
                                synset_similarities1.sort(key=lambda x: x[1], reverse=True)
                                best_synset1 = synset_similarities1[0][0]
                            else:
                                best_synset1 = None
                            
                            # 预测句子2中的词义
                            synset_similarities2 = []
                            for synset in synsets:
                                try:
                                    synset_emb = get_synset_embedding(synset, tokenizer, model)
                                    sim = cosine_similarity([word_embedding2], [synset_emb])[0][0]
                                    synset_similarities2.append((synset, sim))
                                except Exception as e:
                                    continue
                            
                            # 选择相似度最高的词义
                            if synset_similarities2:
                                synset_similarities2.sort(key=lambda x: x[1], reverse=True)
                                best_synset2 = synset_similarities2[0][0]
                            else:
                                best_synset2 = None
                        else:
                            best_synset1 = None
                            best_synset2 = None
                        
                        # 显示结果，保持与传统方法相似的框架
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.write(f'**句子 1**: {sentence1}')
                            # 基于上下文向量方法的预测
                            if best_synset1:
                                st.write(f'**预测词义**: {best_synset1.name()}')
                                st.write(f'**定义**: {best_synset1.definition()}')
                            else:
                                st.warning('未找到合适的词义定义')
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.write(f'**句子 2**: {sentence2}')
                            # 基于上下文向量方法的预测
                            if best_synset2:
                                st.write(f'**预测词义**: {best_synset2.name()}')
                                st.write(f'**定义**: {best_synset2.definition()}')
                            else:
                                st.warning('未找到合适的词义定义')
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning('BERT 模型加载失败，无法使用基于上下文向量的方法')
            except Exception as e:
                st.error(f'处理 BERT 模型时出错: {e}')
        else:
            st.info('未安装 transformers 和 torch 库，无法使用基于 BERT 的上下文向量方法')
            st.info('可以通过运行: pip install --user transformers torch 来安装这些库。')
    st.markdown("</div>", unsafe_allow_html=True)

# 模块二：语义角色标注（SRL）近似提取与可视化
with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 02</div>
                <h3>语义角色标注（SRL）近似提取与可视化</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("通过依存句法和启发式规则近似提取施事者、受事者、时间和地点等角色。")
    render_guide_card(
        "这一部分演示一句话里的谓词和论元是如何组织起来的，也就是谁做了什么、在什么地方、什么时间发生。",
        "先看表格里的 A0、A1、AM-LOC、AM-TMP，再看下面的依存关系图帮助解释为什么系统会这样抽取。",
        "它不是完整的 SRL 模型，而是更适合教学演示的启发式近似版本。",
    )
    
    # 添加模块说明
    with st.expander('📝 模块说明', expanded=True):
        st.markdown('''
        **功能**：使用 spaCy 进行依存句法分析，通过启发式规则近似提取谓词-论元结构。
        
        **原理**：
        - **依存句法分析**：分析句子中词语之间的依存关系。
        - **启发式规则**：基于依存关系和实体识别提取语义角色。
        
        **使用方法**：输入一个句子，系统会提取谓词、施事者、受事者、地点和时间等语义角色，并可视化依存关系。
        ''')
    
    # 句子输入框
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    sentence = st.text_area('输入句子：', value='Apple is manufacturing new smartphones in China this year.', height=100)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 分析按钮
    if st.button('🚀 开始分析', key='srl_analyze'):
        nlp = load_spacy_model()
        doc = nlp(sentence)
        
        # 提取谓词（核心动词）
        predicate = None
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                predicate = token.text
                break
        if not predicate:
            for token in doc:
                if token.pos_ == 'VERB':
                    predicate = token.text
                    break
        
        # 提取 A0（施事者）
        a0 = ''
        for token in doc:
            if token.dep_ == 'nsubj' and (predicate is None or token.head.text == predicate):
                a0 = ' '.join([t.text for t in token.subtree])
                break
        
        # 提取 A1（受事者）
        a1 = ''
        for token in doc:
            if token.dep_ == 'dobj' and (predicate is None or token.head.text == predicate):
                a1 = ' '.join([t.text for t in token.subtree])
                break
        
        # 提取 AM-LOC（地点修饰语）
        am_loc = ''
        for token in doc:
            if token.dep_ == 'prep' and token.head.text == predicate:
                for child in token.children:
                    if child.dep_ == 'pobj' and (child.ent_type_ == 'GPE' or child.ent_type_ == 'LOC'):
                        am_loc = ' '.join([t.text for t in token.subtree])
                        break
                if am_loc:
                    break
        if not am_loc:
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:
                    am_loc = ent.text
                    break
        
        # 提取 AM-TMP（时间修饰语）
        am_tmp = ''
        for token in doc:
            if token.dep_ == 'nmod:tmod' and (predicate is None or token.head.text == predicate):
                am_tmp = ' '.join([t.text for t in token.subtree])
                break
        if not am_tmp:
            for ent in doc.ents:
                if ent.label_ == 'DATE' or ent.label_ == 'TIME':
                    am_tmp = ent.text
                    break
        if not am_tmp:
            for token in doc:
                if token.dep_ == 'prep' and token.text in ['in', 'on', 'at']:
                    for child in token.children:
                        if child.ent_type_ in ['DATE', 'TIME']:
                            am_tmp = ' '.join([t.text for t in token.subtree])
                            break
                    if am_tmp:
                        break
        
        # 显示提取结果
        st.subheader('📊 语义角色提取结果')
        
        # 创建表格
        data = {
            'A0（施事者）': [a0 if a0 else '未识别'],
            '谓词': [predicate if predicate else '未识别'],
            'A1（受事者）': [a1 if a1 else '未识别'],
            'AM-LOC（地点）': [am_loc if am_loc else '未识别'],
            'AM-TMP（时间）': [am_tmp if am_tmp else '未识别']
        }
        df = pd.DataFrame(data)
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 可视化依存关系图
        st.subheader('🗺️ 依存关系图')
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        try:
            from spacy import displacy
            # 美化依存关系图
            options = {
                'compact': False,  # 不使用紧凑模式，避免弧线重叠
                'distance': 160,  # 调整距离，避免弧线重叠
                'color': '#0000ff',  # 蓝色，用于依存关系标签
                'bg': '#f8f9fa',
                'font': 'Arial',
                'add_lemma': True,
                'arrow_stroke': 2,
                'arrow_width': 1.5,
                'word_spacing': 25,
                'token_spacing': 15,
                'collapse_punct': True,
                'collapse_phrases': True,
                'offset_x': 100,  # 增加左侧偏移，避免左侧内容被截断
                'offset_y': 50,  # 增加顶部偏移，避免顶部内容被截断
                'css': '''
                    /* 覆盖默认样式，确保词语和词性显示正确颜色 */
                    .displacy-token {
                        font-family: Arial, sans-serif !important;
                        font-size: 16px !important;
                        font-weight: normal !important;
                        color: #000000 !important;  /* 词语黑色 */
                        background-color: #ffffff !important;
                        padding: 4px 8px !important;
                        border-radius: 4px !important;
                        box-shadow: 1px 1px 3px rgba(0,0,0,0.1) !important;
                    }
                    .displacy-tag {
                        font-family: Arial, sans-serif !important;
                        font-size: 12px !important;
                        font-style: italic !important;  /* 词性斜体 */
                        font-weight: normal !important;
                        color: #0000ff !important;  /* 词性蓝色 */
                        background-color: #f8f9fa !important;
                        padding: 2px 4px !important;
                        border-radius: 3px !important;
                        margin-top: 4px !important;
                        text-align: center !important;
                    }
                    .displacy-dep {
                        font-family: Arial, sans-serif !important;
                        font-size: 11px !important;
                        font-weight: bold !important;  /* 依存关系加粗 */
                        color: #0000ff !important;  /* 依存关系蓝色 */
                        background-color: transparent !important;
                        padding: 0 !important;
                    }
                    .displacy-arrow {
                        stroke: #0000ff !important;  /* 弧线蓝色 */
                        stroke-width: 2px !important;
                        stroke-linecap: round !important;
                    }
                    .displacy-node {
                        transition: all 0.3s ease !important;
                    }
                    .displacy-node:hover {
                        transform: scale(1.05) !important;
                    }
                '''
            }
            html = displacy.render(doc, style='dep', options=options)
            
            # 组合HTML，添加容器样式确保完整显示
            combined_html = '''
            <div style="width: 100%;">
                ''' + html + '''
            </div>
            '''
            st.components.v1.html(combined_html, height=600, scrolling=True)
        except Exception as e:
            st.error(f'渲染依存关系图时出错: {e}')
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
