import umap
import torch
import plotly.express as px


def visualize_embeddings_2d(
    sentence_embeddings,
    hover_name_col=None,
    color_col=None,
    hover_data_cols=None,
    size_col=None,
    width=900,
    height=800,
    return_embeddings=False,
):
    """Use UMAP to project embeddings to 2d. Use Plotly to visualize reduced embeddings
    as a scatterplot. Can display various features of embeddings, such as corresponding item
    name, category, price."""

    if isinstance(sentence_embeddings, torch.Tensor):
        sentence_embeddings = sentence_embeddings.cpu().numpy()

    model = umap.UMAP(n_components=2)
    reduced_embeddings = model.fit_transform(sentence_embeddings)

    fig_kwargs = {
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
    }
    if hover_name_col:
        fig_kwargs.update({"hover_name": hover_name_col})
    if color_col:
        fig_kwargs.update({"color_col": color_col})
    if hover_data_cols:
        fig_kwargs.update({"hover_data": hover_data_cols})
    if size_col:
        fig_kwargs.update({"size": size_col})

    fig = px.scatter(**fig_kwargs)
    fig.update_layout(autosize=False, width=width, height=height)
    fig.show()

    if return_embeddings:
        return reduced_embeddings
