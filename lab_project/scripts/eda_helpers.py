"""Display/export helpers only; the notebook contains the actual EDA queries."""

from html import escape
import json
from pathlib import Path

from IPython.display import HTML, display
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


COLORS = {"source1": "#2563eb", "source2": "#059669", "source3": "#d97706",
          "US": "#2563eb", "India": "#d97706"}


class Report:
    def __init__(self, folder):
        self.folder = Path(folder)
        (self.folder / "tables").mkdir(parents=True, exist_ok=True)
        self.figures = []
        self.table_names = []
        self.findings = []
        # Embedded once: both the notebook and exported report work offline.
        display(HTML('<script type="text/javascript">' + get_plotlyjs() + '</script>'))

    def table(self, name, frame, limit=15):
        frame.write_csv(self.folder / "tables" / f"{name}.csv")
        self.table_names.append(name)
        header = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
        body = "".join("<tr>" + "".join(f"<td>{escape('' if value is None else str(value))}</td>"
                                      for value in row) + "</tr>" for row in frame.head(limit).iter_rows())
        display(HTML(f'<div style="overflow:auto;max-height:520px"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div><p>Showing {min(limit, frame.height)} of {frame.height} rows. Full table: tables/{name}.csv</p>'))

    def chart(self, name, figure, title, x_title=None, y_title=None, **layout):
        figure.update_layout(template="plotly_white", title=title,
                             font=dict(family="Arial, sans-serif", size=13),
                             margin=dict(l=65, r=35, t=85, b=70), height=layout.pop('height', 460),
                             xaxis_title=x_title, yaxis_title=y_title,
                             legend_title_text="", **layout)
        display(HTML(figure.to_html(full_html=False, include_plotlyjs=False,
                                   div_id=f"eda-{name}", config={"responsive": True, "displaylogo": False})))
        self.figures.append({"name": name, "traces": len(figure.data)})

    def note(self, text):
        self.findings.append(text)
        display(HTML(f'<p><strong>Observation:</strong> {escape(text)}</p>'))

    def finish(self, context):
        text = "# Training-data EDA findings\n\n"
        text += "Scope: only the reduced training split. No validation/evaluation records, competition test data, algorithms, or trained models were used.\n\n"
        text += "\n".join(f"- {finding}" for finding in self.findings)
        text += "\n\n## Interpretation limits\n\nThe target pool includes deliberately sampled extra records. Its composition is not an estimate of the original population. Repeated text does not establish shared identity; non-ASCII text is not an error or a language label. Empty ground-truth lists are valid singleton labels.\n"
        (self.folder / "findings.md").write_text(text, encoding="utf-8")
        (self.folder / "eda_run.json").write_text(json.dumps({**context, "charts": self.figures,
              "tables": self.table_names, "findings": self.findings}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bars(frame, category, value, group, order=None):
    figure = go.Figure()
    for name in order or sorted(frame[group].unique().to_list()):
        part = frame.filter(frame[group] == name)
        figure.add_bar(name=name, x=part[category].to_list(), y=part[value].to_list(),
                       marker_color=COLORS.get(name), hovertemplate="%{x}: %{y}<extra>%{fullData.name}</extra>")
    return figure
