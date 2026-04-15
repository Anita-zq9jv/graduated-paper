import os
import openpyxl
from openai import OpenAI
import time

# 设置API客户端
client = OpenAI(
    api_key=os.environ.get(" your key"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

def main():
    # 加载Excel文件
    file_path = "D:/A/2023.xlsx"
    wb = openpyxl.load_workbook(file_path)
    sheet = wb.active

    # 读取固定提示词（E2单元格）
    fixed_prompt = sheet["E2"].value
    if not fixed_prompt:
        print("E2单元格为空，请填写固定提示词！")
        return

    # 获取问题总数
    total_questions = 20
    print(f"总共需要处理 {total_questions} 个问题。")

    for row in range(2, 2 + total_questions):  # 从第2行开始（假设第1行为表头）
        question = sheet[f"A{row}"].value
        if not question:
            print(f"第 {row} 行的问题为空，跳过。")
            continue

        # 构建提示词
        prompt = f"{fixed_prompt}{question}"

        # 调用DeepSeek API获取答案
        try:
            completion = client.chat.completions.create(
                model="deepseek-r1-250120",  # 模型ID
                messages=[
                    {"role": "system", "content": "你是人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            print(f"调用API时出错：{e}")
            answer = "调用API失败"

        # 将答案写入Excel文件（第2列）
        sheet[f"B{row}"].value = answer

        # 输出日志记录
        print(f"Row {row}: {question} — {answer}")

        # 保存Excel文件
        wb.save(file_path)

        # 显示进度
        completed = row - 1
        print(f"进度：已完成 {completed}/{total_questions}")

        # 为避免API请求过快，加入短暂延时
        time.sleep(1)

    print("所有问题已处理完毕！答案已写入Excel文件。")

if __name__ == "__main__":
    main()
