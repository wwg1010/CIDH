import matplotlib.pyplot as plt
import json
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties( size=18,weight='bold')

from matplotlib.font_manager import FontProperties
# Precision Recall Curve data
# pr_data = {
#     # "ISDH_m": "../log/alexnet/ISDH_mirflickr_32.json",
#     "ISDH_m": "../log/alexnet/ISDH_coco_32.json",
# }
pr_data = {
    # "DPSH1": "../log/UCMD/DPSH_UCMD_32.json",
    # "DPSH2": "../log/UCMD/DPSH_UCMD_48.json",
    # "DPSH3": "../log/UCMD/DPSH_UCMD_64.json",
    # "DPSH4": "../log/UCMD/DPSH_UCMD_96.json",
    # "DPSH1": "../log/AID/DPSH_AID_32.json",
    # "DPSH2": "../log/AID/DPSH_AID_48.json",
    # "DPSH3": "../log/AID/DPSH_AID_64.json",
    # "DPSH4": "../log/AID/DPSH_AID_96.json",
    "DPSH1": "../log/WHURS/DPSH_WHURS_32.json",
    "DPSH2": "../log/WHURS/DPSH_WHURS_48.json",
    "DPSH3": "../log/WHURS/DPSH_WHURS_64.json",
    "DPSH4": "../log/WHURS/DPSH_WHURS_96.json",
}
N = 100
# N = -1
for key in pr_data:
    path = pr_data[key]
    pr_data[key] = json.load(open(path))


# markers = "DdsPvo*xH1234h"
markers = "...................."
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

plt.figure(figsize=(16, 4))
plt.subplot(131)

for method in pr_data:
    P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    print(len(P))
    print(len(R))
    plt.plot(R, P, linestyle="-", marker=method2marker[method], label=method)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.subplot(132)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, R,  marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range)+1)
plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('Returned samples',fontproperties=font)
plt.ylabel('Recall',fontproperties=font)
plt.legend()

plt.subplot(133)
N1 = 10
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N1],pr_data[method]["R"][:N1],pr_data[method]["index"][:N1]
    plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range)+1)
plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('Returned samples',fontproperties=font)
plt.ylabel('Precision',fontproperties=font)
plt.legend()
plt.savefig("pr.png")
plt.show()
