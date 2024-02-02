from os import path
import jieba
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
from PIL import Image

if __name__ == '__main__':

    # 获取路径
    d = path.dirname(__file__)
    # 下面四个自行变换
    TXT_path = path.join(d, 'doc//浪潮之巅.txt')  # 文本
    MASK_path = path.join(d, 'pic//雾雨魔理沙.png')  # 图片
    STOPWORDS_path = path.join(d, r'doc//stopwords_cn.txt')  # 停用词
    FONT_path = path.join(d, 'font//msyh.ttf')  # 字体
    USERDICT_path = path.join(d, 'doc//自定义词组.txt')

    # 找到txt文件
    text = open(TXT_path, encoding='UTF-8').read()
    # 找到mask文件
    mask = np.array(Image.open(MASK_path))
    # 导入自定义词典
    jieba.load_userdict(USERDICT_path)
    # 提取背景颜色
    bg_color = ImageColorGenerator(mask, default_color=None)

    # 若是中文文本，则先进行分词操作
    # cut_all是分词模式，True是全模式，False是精准模式，默认False
    wordTemp = jieba.lcut(text, cut_all=True)

    words = []

    # 设定停用词表
    stopword = [line.strip() for line in open(STOPWORDS_path, 'r', encoding='UTF-8').readlines()]

    # 载入词
    for w in wordTemp:
        if w not in stopword:
            words.append(w)

    # 去除空格
    words = [item.strip() for item in words if item.strip() != '']

    # 去停用词之后的词频统计结果
    frequency = dict(Counter(words))

    # 记录词频，用于调试
    # with open('词频统计.txt', 'w', encoding='UTF-8') as fw:
    #     for k, v in frequency.items():
    #         fw.write("%s,%d\n" % (k, v))

    wc = WordCloud(background_color="white",  # 设置背景颜色
                   max_words=500,  # 词云显示的最大词数
                   mask=mask,  # 设置背景图片
                   font_path=FONT_path,  # 兼容中文字体，不然中文会显示乱码
                   # contour_color='orange',# 边框颜色
                   # contour_width=2,#边框粗细
                   color_func=bg_color,  # 背景颜色
                   )

    # 词频生成词云
    wc.generate_from_frequencies(frequency)
    # 文本生成词云
    # wc.generate(words)

    wc.to_file('output.png')
