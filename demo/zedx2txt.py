import os
import json
import chardet
import html2text
from tqdm import tqdm
from bs4 import BeautifulSoup
import sys
import re


def extract_html(dir_path):
    files = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        if os.path.isdir(file_path):
            # If this is a directory, process it recursively
            files.extend(extract_html(file_path))
        elif file_name.endswith(".html"):
            files.append(file_path)
    return files


def addtag(soup,head,signst,signed,sttag,edtag):
    # Find table
    for span in soup.find_all(**head) :
        try:
            start_tag = soup.new_tag(sttag)  
            start_tag.string = signst

            end_tag = soup.new_tag(edtag)  
            end_tag.string = signed

            span.insert_before(start_tag)
            span.insert_after(end_tag)
        except:
            continue
    return 0

def deltag(soup,head):
    found=False
    for span in soup.find_all(**head) :
        try:
            span.replace_with('')  
            found=True
        except:
            continue
    return found

def replacetext(soup,head,text):
    for span in soup.find_all(**head) :
        try:
            en, cn = title.split("--")
            span.string = f"{text}{span.string}"
        except:
            continue

def parse_zedx(input_path, output_path, error_path):
    os.makedirs(output_path, exist_ok=True)
    files = extract_html(os.path.join(input_path, "documents"))
    error_files = []
    for file in tqdm(files, desc=input_path):
        # 跳过一些空文件
        if os.stat(file).st_size == 0:
            continue
        with open(file, "r") as f:
            try:
                html_content = f.read()
            except UnicodeDecodeError:  # 一些文件的编码不是utf-8
                rawdata = open(file, "rb").read()
                result = chardet.detect(rawdata)
                encoding = result["encoding"]
                with open(file, "r", encoding=encoding):
                    html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # 找到所有class为"xref gxref"的span元素，对zedx缩略语进行补充
        for span in soup.find_all("span", class_="xref gxref"):
            title = span.get("title")
            if title:
                try:
                    en, cn = title.split("--")
                    span.string = f"{span.string}({en}, {cn})"
                except:
                    span.string = f"{span.string}({title})"
                    error_files.append(file)

        addlabel=True
        if addlabel:
	    # specific sign
            # 标记table
            # addtag(soup,{"name":"table"},"_TS","_TE",sttag='div',edtag='div')
            # 标记title 
            addtag(soup,{"name":"div","data-dtd-path":"ztetopic/title"},"#","",sttag='span',edtag='span')
            addtag(soup,{"name":"div","data-dtd-path":"zteconcept/title"},"#","",sttag='span',edtag='span')
            addtag(soup,{"name":"div","data-dtd-path":"dita/ztetopic/title"},"#","",sttag='span',edtag='span')
            # 标记subtitle
            addtag(soup,{"name":"div","data-dtd-path":"dita/ztetopic/ztetopic/title"},"##","",sttag='span',edtag='span')
            addtag(soup,{"name":"div","data-dtd-path":"ztetopic/body/section/title"},"##","",sttag='span',edtag='span')
            addtag(soup,{"name":"div","data-dtd-path":"zteconcept/conbody/section/title"},"##","",sttag='span',edtag='span')
            # 标记说明
            addtag(soup,{"name":"div","class":"note note-note"},"",")",sttag='span',edtag='div')
            # 标记一个子章节的开始：
            addtag(soup,{"name":"div","class":"sectiondiv"},"###","",sttag='span',edtag='span')

        html = str(soup)
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        text = h.handle(html)

        if addlabel:
	    # Post processing
            # 去掉多余空行
            #text = re.sub('\n\n+','\n',text)
            # 标题处理
            #text=text.replace('## ','~ ')
            #text=text.replace('# ','$ ')
            text=text.replace('###\n\n',"### ")
            text=text.replace('##\n\n',"## ")
            text=text.replace('#\n\n',"# ")
            # 去掉^图
            text = re.sub('\n图\d+.*\n','\n',text)
            # 说明，特殊处理
            text = re.sub('说明：\n','(说明:',text)
            text = re.sub("\n\)",")",text)
        #try:
        #    if '子主题' in text.split('\n\n')[1]:
        #        continue
        #except Exception as e:
        #    continue
        if len(re.sub('[a-zA-Z\s]+','',text,re.I))<50:
            continue


        path = os.path.normpath(file)
        paths = path.split(os.sep)
        try:
            paths.remove("topics")
        except ValueError:
            pass
        paths = paths[paths.index("documents") + 1 :]
        save_path = f"{os.path.splitext('/'.join(paths))[0]}.md"
        final_save_path = os.path.join(output_path, save_path)
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        with open(final_save_path, "w", encoding="utf-8") as f:
            f.write(text)
       
        # Give a new summary file

        if False:
            html = str(soup)
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.body_width = 0
            text = h.handle(html)
            found = deltag(soup,{"name":"table"})
            #if not found:
            #    continue

            html = str(soup)
            text = h.handle(html)
            final_save_path = os.path.join(output_path+'_sumary', save_path)
            os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
            with open(final_save_path, "w", encoding="utf-8") as f:
                f.write(text)

    if len(error_files) > 0:
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_files, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_path = f"./dataset/zedxs/{sys.argv[1]}"
    output_path = f"dataset/zedxs/text/{sys.argv[1]}"
    error_path = f"dataset/zedxs/text/{sys.argv[1]}.json"
    parse_zedx(input_path, output_path, error_path)
