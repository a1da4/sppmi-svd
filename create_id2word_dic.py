from collections import Counter
import sys
import re

# min_count: これより多く出現した単語を考慮
# del_kana_num_alphabet: ひらがな、数字、英語で始まる語を削除
min_count = 20
del_kana_num_alphabet = False

f_name = sys.argv
f_name = f_name[1]

sent_all = []

with open(f_name) as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(" ")
        # 記号を削除
        line = [re.sub(r"「|」|『|』|、|\n|\s|！|？|（|）|〔|〕|【|】|《|》|〈|〉|［|］|”|“|’|゛|〃|ゝ|ゞ|ヽ|ヾ|．|，|‥|…|〜|＝|・|；|：|＿|−|‐|—|-|─|／|●|◉|◯|〇|◎|○|△|▲|▼|▽|◆|□|■|◇|★|☆|†|〒|＊|×|＋","", l) for l in line if len(re.sub(r"「|」|『|』|、|\n|\s|！|？|（|）|〔|〕|【|】|《|》|〈|〉|［|］|”|“|’|゛|〃|ゝ|ゞ|ヽ|ヾ|．|，|‥|…|〜|＝|・|；|：|＿|−|‐|—|-|─|／|●|◉|◯|〇|◎|○|△|▲|▼|▽|◆|□|■|◇|★|☆|†|〒|＊|×|＋", "", l)) > 0]
        # ひらがな、数字、英字で始まるものを削除
        if del_kana_num_alphabet:
            line = [re.sub(r"^[ぁ-んa-zA-Zａ-ｚＡ-Ｚ0-9]+", "", l) for l in line if len(re.sub(r"^[a-zA-Z]+$", "", l)) > 0]
        if len(line) == 0:
            continue
        sent_all.extend(line)

count_dic = Counter(sent_all)
id=0
with open("id_to_word.txt", mode="w") as f:
for k, v in count_dic.most_common():
    if v < min_count:
        break
    f.write("{}\t{}".format(id,k))
    id+=1
