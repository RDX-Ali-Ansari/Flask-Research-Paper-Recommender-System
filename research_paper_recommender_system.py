import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

@st.cache_data
def load_data():
    df = pd.read_json("arxivData.json")
    df.drop(columns=['day', 'month', 'year'], inplace=True)

    # Convert JSON fields to lists
    df['author'] = df['author'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
    df['tag'] = df['tag'].apply(lambda x: [i['term'] for i in ast.literal_eval(x)])
    df['link'] = df['link'].apply(lambda x: ast.literal_eval(x)[0]['href'] if pd.notnull(x) else "")

    # Clean author names
    df['author'] = df['author'].apply(lambda L: [i.replace(" ", "") for i in L])

    # Prepare tags
    df['summary'] = df['summary'].apply(lambda x: x.split())
    df['tags'] = df.apply(lambda row: " ".join(row['summary'] + row['tag'] + row['author']), axis=1)

    # Stemming
    ps = PorterStemmer()
    df['tags'] = df['tags'].apply(lambda text: " ".join([ps.stem(i) for i in text.split()]))

    # ✅ Return only safe columns
    return df[['title', 'link', 'tags']]

# ----- Create Vectors & Similarity Matrix -----
@st.cache_resource
def compute_similarity(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vector)
    return similarity

# ----- Recommendation Logic -----
def recommend(paper_title, df, similarity):
    if paper_title not in df['title'].values:
        return []

    index = df[df['title'] == paper_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
    recommendations = []

    for i in distances:
        rec_title = df.iloc[i[0]].title
        rec_link = df.iloc[i[0]].link
        recommendations.append((rec_title, rec_link))

    return recommendations

# ----- Streamlit App UI -----
st.set_page_config(page_title="Research Paper Recommender", layout="wide")
st.title("📚 Research Paper Recommender System")
st.markdown("Get personalized suggestions for research papers based on a selected paper.")

df = load_data()
similarity = compute_similarity(df)

paper_titles = df['title'].sort_values().unique()

selected_title = st.selectbox("Select a Research Paper:", paper_titles)

if st.button("Recommend Similar Papers"):
    with st.spinner("🔍 Finding similar papers..."):
        results = recommend(selected_title, df, similarity)

        if results:
            st.success("Top 5 Recommended Papers:")
            for idx, (title, link) in enumerate(results, 1):
                st.markdown(f"**{idx}. [{title}]({link})**")
        else:
            st.warning("No recommendations found.")
