'''
# Conduct EDA
# Identify features
# Do pre-processing
# Fit MLP and RF
# Compare output
'''

# ----------------------------------------
### Data import ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib import rcParams

# Look at dataset
data = pd.read_csv('../Data/data.csv', header=0)
print(data.head)
print(data.columns)
'''
['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat',
       'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs',
       'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag',
       'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
       'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id',
       'made']
'''

# Get column datatypes and summaries
print(data.dtypes)

# Drop rows which were intended for Kaggle competition
# These rows are missing information about whether the shot was made or not
#
data1 = data.dropna()
#data1 = data1[(data1['shot_made_flag'] == 1) | (data1['shot_made_flag'] == 0)]
data1['made'] = np.where(data1['shot_made_flag'] == 1.0, 'Made', 'Missed')

# ----------------------------------------
### EDA ###


'''
Ideas for EDA. Don't need to do all of them.
#1. Find kobe average - just realized that only 2FG and 3FG are only included in our dataset
#2. Which action type worked/failed the most
#3. From which zone area did he score/fail the most
#4. From which shot zone range did he score/fail the most
#5. Against which team did he best/worst perform
#6. Scoring avg per period
#7. Do analysis based on minutes and seconds remaining
'''

# Shot chart - overlay zone on court
pd.pivot_table(data1, index=['shot_type'],values=['made'], columns=['period'], aggfunc='count')
pd.pivot_table(data1, index=['shot_zone_area'],values=['shot_made_flag'], columns=['made'], aggfunc='count')


# analysis by game time - minutes and seconds


# Function to draw court and plot Bryant's shots
# Function obtained from external source
# http://savvastjortjoglou.com/nba-shot-sharts.html
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

# Example of empty court
#plt.figure(figsize=(12,11))
#draw_court(outer_lines=True)
#plt.xlim(-300,300)
#plt.ylim(-100,500)
#plt.show()

made_array = np.where(data1['made'] == 'Made', 'green', 'red')

plt.figure(figsize=(12,11))
plt.scatter(data1.loc_x, data1.loc_y, c=made_array)
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
plt.savefig('../Graphs/ShotChart_v1_draft.jpeg', dpi=300, bbox_inches='tight')
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
plt.scatter(data1.loc_x, data1.loc_y, c=data1['shot_zone_area'].map(shot_zone_dict))
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
plt.savefig('../Graphs/ShotChart_v2_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()

shot_zone_basic_dict = {
    'Mid-Range':'blue',
    'Restricted Area':'red',
    'In The Paint (Non-RA)':'orange',
    'Above the Break 3':'green',
    'Right Corner 3':'lightgreen',
    'Backcourt':'grey',
    'Left Corner 3':'greenyellow'

}

plt.figure(figsize=(12,11))
plt.scatter(data1.loc_x, data1.loc_y, c=data1['shot_zone_basic'].map(shot_zone_basic_dict))
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
plt.savefig('../Graphs/ShotChart_v3_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()

'''
# Decide if we still need this
ax = data1.plot.hexbin(x='loc_x',
                    y='loc_y',
                    C='shot_made_flag',
                    reduce_C_function=np.mean,
                    gridsize=30,
                    cmap='RdYlGn')
plt.savefig('../Graphs/ShotChart_v2_draft.jpeg', dpi=300, bbox_inches='tight')
#plt.figure(figsize=(12,11))
#draw_court(outer_lines=True)
#plt.xlim(300,-300)
#plt.ylim(-100,500)
#plt.show()
'''

# Add hex to aggregate shot values
def create_court(ax, color):
    # Short corner 3PT lines
    ax.plot([-220, -220], [0, 140], linewidth=2, color=color)
    ax.plot([220, 220], [0, 140], linewidth=2, color=color)

    # 3PT Arc
    ax.add_artist(Arc((0, 140), 440, 315, theta1=0, theta2=180, facecolor='none', edgecolor=color, lw=2))

    # Lane and Key
    ax.plot([-80, -80], [0, 190], linewidth=2, color=color)
    ax.plot([80, 80], [0, 190], linewidth=2, color=color)
    ax.plot([-60, -60], [0, 190], linewidth=2, color=color)
    ax.plot([60, 60], [0, 190], linewidth=2, color=color)
    ax.plot([-80, 80], [190, 190], linewidth=2, color=color)
    ax.add_artist(Circle((0, 190), 60, facecolor='none', edgecolor=color, lw=2))

    # Rim
    ax.add_artist(Circle((0, 60), 15, facecolor='none', edgecolor=color, lw=2))

    # Backboard
    ax.plot([-30, 30], [40, 40], linewidth=2, color=color)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set axis limits
    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 470)

    return ax

rcParams['font.family'] = 'Avenir'
rcParams['font.size'] = 18
rcParams['axes.linewidth'] = 2


# Create figure and axes
fig = plt.figure(figsize=(4*2, 3.76*2)) # Was 4, 3.76
ax = fig.add_axes([0, 0, 1, 1])

# Draw court
ax = create_court(ax, 'black')

# Plot hexbin of shots
ax.hexbin(data1['loc_x'], data1['loc_y'] + 60,
          gridsize=(50, 50),
          extent=(-300, 300, 0, 940),
          bins=12,
          C=data1['shot_made_flag'],
          reduce_C_function=np.mean,
          cmap='RdYlGn')

# Annotate player name and season
ax.text(0, 1.05, 'Kobe Bryant\nCareer Shooting', transform=ax.transAxes, ha='left', va='baseline')

# Save and show figure
plt.savefig('../Graphs/ShotChart_v4_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# Add avg field goal percentage by distance - do (1) bins and (2) continuous line

# Look at shot zone chart for teams in which he had highest field goal percentage and lowest field goal percentage
# Use effective field goal percentage
# https://www.basketball-reference.com/about/glossary.html
