import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    fileLoc = r"C:\Users\Joshh\Projects\Stocks\Data\combined_trade_data - Copy.csv"
    df = pl.read_csv(fileLoc)
    return (df,)


@app.cell
def _(df):
    df.head(20)
    return


if __name__ == "__main__":
    app.run()
