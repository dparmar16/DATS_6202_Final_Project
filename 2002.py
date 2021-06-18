data2002=data1[data1['season']=='2001-02']
# made_array2002 = np.where(data2002['made'] == 'Made', 'green', 'red')
cmade={'Made':'green','Missed':'red'}
plt.figure(figsize=(12,11))
plt.scatter(data2002.loc[data2002['opponent']=='MEM','loc_x'], data2002.loc[data2002['opponent']=='MEM','loc_y'], c=data2002.loc[data2002['opponent']=='MEM','made'].map(cmade))
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
# plt.savefig('../Graphs/ShotChart_v1_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()

print(data1.shot_zone_area.unique())

shot_zone_dict = {'Left Side(L)': 'red',
                  'Left Side Center(LC)': 'orange',
                  'Center(C)': 'purple',
                  'Right Side Center(RC)': 'green',
                  'Right Side(R)': 'blue',
                  'Back Court(BC)': 'grey'
                  }


plt.figure(figsize=(12,11))
plt.scatter(data2002.loc[data2002['opponent']=='MEM','loc_x'], data2002.loc[data2002['opponent']=='MEM','loc_y'], c=data2002.loc[data2002['opponent']=='MEM','shot_zone_area'].map(shot_zone_dict))
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
# plt.savefig('../Graphs/ShotChart_v2_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()