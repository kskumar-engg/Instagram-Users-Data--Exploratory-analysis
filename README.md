
# Instagram-Users-Data--Exploratory-analysis
Module 1:  
Aims to identify fake influencers & emerging influencers:  
&nbsp;&nbsp;&nbsp;&nbsp;i) High followers and followers growth  
&nbsp;&nbsp;&nbsp;&nbsp;ii)Less engagement rate.  
     There is a possibility such user is a celebrity/famous to have such growth. This is handled in module 2 to exclude those users and only identify phantom influencers.

## Documentation



**1.Background of the Dataset**: This dataset is about the Instagram social network, specifically focused on Influence Maximization (IM). The data was collected from Instagram during April to May 2020. It was sourced from the followers of 24 private universities in Malaysia using the Instagram API and various third-party Instagram websites. The dataset represents mostly Malaysian Instagram users with a network of their connections (followers and followees). It has detailed statistics such as engagement rate, number of posts, followers growth percentage, and more for each user. 

**2.Literature Review**: Papers referred: Title: Influence Maximization Diffusion Models Based On Engagement and Activeness on Instagram Summary: This paper discusses the role of influencers on social media, specifically Instagram. It identifies the limitations of existing influence maximization models and introduces three new diffusion models. These models consider user engagement and activeness, which were often overlooked in previous models. The new models were tested using the provided dataset.

**3.Experimental Dataset**  
**a.Exploratory Data Analysis**: The dataset comprises user statistics and network connections. Features like engagement rate, followers’ growth percentage, and number of posts offer insights into user activity and influence on Instagram. As the graph is massive, it is impossible to render and visualize, hence a subgraph has been plotted with 1000 nodes. However we found useful metrics like giant component *(fig3 @charts.ipynb)* for both main and subgraph.

**Local, Global clustering coefficients and Clusters** *(fig1 @charts.ipynb)*: Top 5 local clustering coefficients nodes i.e nodes that are maximum connected are returned and clusters are found using label propagation algorithm.

**Selected Centrality Measures:**  
    **Betweenness Centrality** *(fig5 @charts.ipynb)*: We have chosen betweenness centrality to identify users who act as crucial intermediaries or bridges in the network. We are interested in pinpointing users     who serve as critical bridges between different parts of the Instagram network. These users can be essential for marketing and information dissemination.  
    **Katz Centrality** *(fig4 @charts.ipynb)*: Katz Centrality is selected to consider both direct connections and connections of connections, capturing users who are well-connected themselves and linked to     other influential users. Users who are connected to other influential users, even if they don't have the most followers, can play a significant role in spreading information and influencing the network.  
    **PageRank Centrality** *(fig2 @charts.ipynb)*: PageRank centrality is included to identify users highly regarded by other influential users. This measure helps identify authorities or experts within the     network. PageRank is included as it reflects the quality of incoming links and the users reputation within the network.

**Excluded Centrality Measures:**  
    **Degree Centrality**: Degree centrality is not used as the sole measure because it solely counts the number of connections without considering their quality. In a large network like Instagram, many     users may have high degrees due to the number of followers or followees, but this doesn't necessarily indicate their true influence.  
    **Eigenvector Centrality**: Eigenvector centrality is omitted because it assumes that influential nodes are connected to other influential nodes. This assumption may not hold in all cases, potentially missing users with isolated but substantial influence.  
    **Closeness Centrality**: Closeness centrality is not applied due to the computational challenges of calculating the shortest paths for all users in a large network. Moreover, it may not accurately     reflect influence if a user's connections are concentrated in specific network regions.  

**b.Statistics and Measures:**  
    Total Users (nodes): 70,409  
    Total Connections (edges): 1,007,107  
    ●The first data set has posts and follower data for each node. This followers information gives the degree of each node, but the edges connected to the node are not known.  
    ●The second and third data sets have source and target nodes if activated, i.e if the calculated assumptions match with nodes that were actually influenced.  
    **Distribution of Followers to Following Ratio** *(fig1 @charts.ipynb)*: A very few users are having high followers and following minimum users. Histograms show thousands of users are regular users.  
    ●**Impact of users engagement attracting non followers** *(fig2 @charts.ipynb)*: How engaging users are increasing the influence range by attracting outsiders . The graph shows better engagement results in     better Here both engagement rate and grade are considered. Engagement Rate is (likes+comments) / followers. The engagement grade is rate/followers and essential as users with low followers always     have higher engagement rate.  
    ●**Fake influencers detection** *(fig3 @charts.ipynb)*: Users with High( Followers, Followers Growth) and less(Engagement Rate) indicate users with bots as followers/fake followers. The graph plotted     doesn’t show any such user as engagement rate decent in the existing data set. If present, can be easily spotted from the plot.  
    ●**Bubble chart(Scatter Plot)**: **Emerging/Budding influencers** *(fig4 @charts.ipynb)*: - Low followers, high growth rate with high outsiders percentage indicates that the users post are reaching a wide     audience. The bubble scatter plot shows users with less followers with high outsider influence(yellow bubble) have high follower growth and this relationship between high outsider influence and     follower growth seems directly proportional to each other.  
## FAQ

#### What is the dataset used?

https://www.kaggle.com/datasets/krpurba/im-instagram-70k/data



