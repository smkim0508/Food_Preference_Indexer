import pandas as pd

def group_reviews_by_user(file_path, user_col='user_id', review_col='business_id'):
    # load excel
    df = pd.read_excel(file_path)

    # group by user_id
    grouped = df.groupby(user_col)[review_col].apply(list).reset_index()

    # add column for review count
    grouped['review_count'] = grouped[review_col].apply(len)

     # filter users with more than 10 reviews
    filtered = grouped[grouped[review_col].apply(len) > 10]

    return filtered

grouped_reviews = group_reviews_by_user('yelp.xlsx')
print(grouped_reviews)