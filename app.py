import logging
import pickle
import jieba
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS, cross_origin
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
import re


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.basicConfig(level=logging.INFO)

logging.getLogger('flask_cors').level = logging.DEBUG

CORS(app, resources={r"/api/*": {"origins": "http://172.18.34.25:80"}})

# 学习出版社会议演示test cases
titles = ["高秀丽与田双阳等机动车交通事故责任纠纷一审民事判决书",
          "冯亚泉与崔耀方、刘红涛机动车交通事故责任纠纷一审民事判决书",
          "程步东诉元红新、汤阴县诚信机动车驾驶员培训学校、信达财产保险股份有限公司安阳中心支公司机动车交通事故责任纠纷案一审民事判决书",
          "李某甲与任飞、岑永坤、中华联合财产保险股份有限公司雅安中心支公司机动车交通事故责任纠纷一案民事判决书"]
labels = ["保险公司列为被告、机动车所有人与使用人不一致、伤残、受害人住院、医疗费、残疾赔偿金、精神抚慰金、被告全部责任、未提起过刑事附带民事诉讼",
          "保险公司列为被告、机动车所有人与使用人不一致、未投保交强险、未投保商业三者险、多辆机动车致人损害、驾驶人逃逸、驾驶人酒驾、伤残、受害人住院、医疗费、残疾赔偿金、被告全部责任、未提起过刑事附带民事诉讼",
          "保险公司列为被告、工作人员驾驶机动车、机动车所有人与使用人不一致、培训活动中出现交通事故、伤残、受害人有过错、受害人住院、医疗费、残疾赔偿金、精神抚慰金、被告主要责任、未提起过刑事附带民事诉讼",
          "保险公司列为被告、机动车所有人与使用人不一致、伤残、受害人有过错、受害人住院、医疗费、残疾赔偿金、精神抚慰金、被告主要责任、未提起过刑事附带民事诉讼"]


@app.route('/')
def hello_world():
    return "<h1 style='color:blue'>同案智推API终端</h1>"


@app.before_first_request
def load_dictionary():
    jieba.load_userdict('resources/solr分词--包含法律词汇.txt')


@app.route('/api/predict', methods=['POST'])
@cross_origin(origin='172.18.34.25', headers=['Content-Type'])
def get_prediction():
    """
        学习出版社会议演示
    """
    title = get_title(request.data)
    tags = []
    flag = 0
    for i in range(len(titles)):
        if title == titles[i]:
            tags = labels[i].split("、")
            flag = 1
            break

    # check文件是否在演示案例中
    if flag == 1:
        return jsonify(tags=tags)
    else:
        # 预测责任方式
        vect_path = 'resources/vectorizer.pkl'
        vectorizer = read_obj(vect_path)

        model_path = 'resources/xgboost.pkl'
        clf = read_obj(model_path)

        contents = []
        txt = get_text(request.data)
        segments = cut_words(txt)
        contents.append(segments)
        tdm = vectorizer.transform(contents).toarray()
        y_pred = clf.predict(tdm)
        label_1 = y_pred[0]

        # 预测多辆机动车致人伤害
        feature_path = 'resources/feature.pkl'
        vectorizer_2 = CountVectorizer(decode_error="replace", min_df=0.005, vocabulary=read_obj(feature_path))

        transformer_path = 'resources/tfidftransformer.pkl'
        tfidftransformer = read_obj(transformer_path)

        model_path = 'resources/xgboost_2.pkl'
        clf_2 = read_obj(model_path)

        contents_2 = []
        txt_2 = get_text_2(request.data)
        segments_2 = cut_words_2(txt_2)
        contents_2.append(segments_2)
        tdm_2 = tfidftransformer.transform(vectorizer_2.transform(contents_2)).toarray()
        y_pred_2 = clf_2.predict(tdm_2)
        label_2 = '是' if y_pred_2[0] == 1 else '否'

        return jsonify(label_1=label_1, label_2=label_2)


@app.route('/api/exception')
def get_exception():
    raise Exception('example')


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request. %s', e)
    return "An internal error occured", 500


def read_obj(path):
    file_obj = open(path, "rb")
    obj = pickle.load(file_obj)
    file_obj.close()
    return obj


def get_title(xml_string):
    title = ""
    try:
        root = ET.fromstring(xml_string)
        nodes = root[0].findall("标题")
        if nodes:
            title += nodes[0].text
    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)
    finally:
        return title


def get_text(xml_string):
    txt = ''
    try:
        root = ET.fromstring(xml_string)
        paragraphs = ['本院认为', '本院查明', '原告诉称', '审理经过']
        for paragraph in paragraphs:
            nodes = root[0].findall(paragraph)
            if nodes:
                txt += nodes[0].text
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        return xml_string
    return txt


def get_text_2(xml_string):
    txt = ''
    try:
        root = ET.fromstring(xml_string)
        paragraphs = ['原告诉称', '被告辩称', '被告诉称', '本院查明', '本院认为', '当事人信息']
        for paragraph in paragraphs:
            nodes = root[0].findall(paragraph)
            if nodes:
                paragraph_txt = nodes[0].text
                # 查看段落是否以句号结尾，若不是则加上句号
                if paragraph_txt[-1] != '。':
                    txt += paragraph_txt + '。'
                else:
                    txt += paragraph_txt
    except Exception:
        return xml_string
    return txt


def cut_words(txt):
    return " ".join(jieba.cut(txt))


# 清洗文本，返回切词后的list
def cut_words_2(txt):
    txt = str(txt).strip()
    # 第一步：去掉特殊符号
    reg0 = '&amp;|&rdquo;|&ldquo;|&times;|rdquo;|ldquo;|times;|&bull;'
    pattern0 = re.compile(reg0)
    txt = re.sub(pattern0, '', txt)

    # 第二步：标记车牌
    # 车牌号正则
    reg1 = '[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1}'
    reg2 = '[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]{1}.{1,3}[×xX＊*☆★○]{3,5}'
    pattern1 = re.compile(reg1)
    pattern2 = re.compile(reg2)

    txt_tmp = txt
    dic = dict()
    tmp_list = re.findall(pattern1, txt_tmp)
    for v in tmp_list:
        if v not in dic.keys():
            dic[v] = len(v)

    tmp_list = re.findall(pattern2, txt_tmp)
    for v in tmp_list:
        if v not in dic.keys():
            dic[v] = len(v)

    # 车牌长度排序，降序
    sorted_key_list = sorted(dic.items(), key=lambda e: e[1], reverse=True)
    i = 1
    for tp in sorted_key_list:
        string = 'CHEPAI' + str(i)
        txt = txt.replace(str(tp[0]), string)
        i += 1
    lst = cut_words_helper(txt)
    return lst


# 对列表进行分词并用空格连接
def cut_words_helper(cont):
    text = ""
    stopwords_list = get_stopwords()
    word_list = list(jieba.cut(cont, cut_all=False))
    for word in word_list:
        if word not in stopwords_list and word != '\r\n':
            text += word
            text += ' '
    return text


def get_stopwords():
    stopwords_path = 'resources/stop_words_ch.txt'
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords_list = f.readlines()
    return stopwords_list


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
