import subprocess
import json

N = 10
s = {}
score = [0] * N

with open('./game_manager/my_param.py') as fr:
    param = fr.read()

for i in range(N):
    #cmd = 'python start.py -m sample -l 2 -d 1 -r ' + str(i)
    cmd = 'python start.py -l 2 -d 1 -r ' + str(i)
    result = subprocess.run(cmd, shell=True)
    with open('./result.json') as f:
        s = json.loads(f.read())
    score[i] = int(s["judge_info"]["score"])

with open('./my_log/0.log', mode='w') as fw:
    for i in range(N):
        sw = str(i) + ": " + str(score[i]) + "\n"
        fw.write(sw)
        print(i, ": ", score[i])
    sw = "ave: " + str(sum(score)/len(score)) + "\n"
    fw.write(sw)
    print("ave: ", sum(score)/len(score))
    fw.write(param)

