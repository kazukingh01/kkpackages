import pandas as pd
from typing import List, Tuple
from jinja2 import Template


__all__ = [
    "create_chart",
    "create_datatables",
    "create_bs4table",
]


def create_chart(
    df: pd.DataFrame, bindto: str, data_x: str, data_x_type: str, x_label: str,
    data_y1: List[str],      data_y1_type: dict,      data_y1_name: dict,      y1_label: str,      y1_range: (float, float, ),
    data_y2: List[str]=None, data_y2_type: dict=None, data_y2_name: dict=None, y2_label: str=None, y2_range: (float, float, )=None, 
    lines: List[dict]=[], colors: dict=None, padding: dict=None,
) -> str:
    """
    dataframe を input として c3 chart を書く
    Params::
        df: input dataframe
        data_x: x軸のカラム名
        data_x_type: x軸のタイプ. str, 
    """

    # template に埋め込む変数を定義していく
    tpl_data_x = {}
    tpl_data_x["name"]  = x_label
    tpl_data_x["type"]  = data_x_type
    tpl_data_x["data"]  = df[data_x].tolist()
    tpl_data_y_op = {}
    tpl_data_y_op["type"]  = ""
    tpl_data_y_op["groups"] = []
    tpl_data_y = []
    for colname in data_y1:
        dictwk = {}
        dictwk["name"] = data_y1_name[colname]
        if data_y1_type.get(colname) is not None:
            if   data_y1_type.get(colname) == "bar_group":
                tpl_data_y_op["type"] = "bar"
                tpl_data_y_op["groups"].append(data_y1_name[colname])
                dictwk["type"] = ''
            elif data_y1_type.get(colname) == "bar":
                tpl_data_y_op["type"] = "bar"
                dictwk["type"] = ''
            else:
                dictwk["type"] = data_y1_type.get(colname)
        else:
            dictwk["type"] = ''
        dictwk["color"]  = colors.get(colname) if colors.get(colname) is not None else ''
        dictwk["data"]   = df[colname].tolist()
        dictwk["y1ory2"] = "y"
        tpl_data_y.append(dictwk)
    if data_y2 is not None:
        for colname in data_y2:
            dictwk = {}
            dictwk["name"]   = data_y2_name[colname]
            dictwk["type"]   = data_y2_type.get(colname) if data_y2_type.get(colname) is not None else ''
            dictwk["color"]  = colors.get(colname) if colors.get(colname) is not None else ''
            dictwk["data"]   = df[colname].tolist()
            dictwk["y1ory2"] = "y2"
            tpl_data_y.append(dictwk)
    if len(tpl_data_y_op["groups"]) > 0:
        tpl_data_y_op["groups"] = "'" + "','".join(tpl_data_y_op["groups"]) + "'"
    else:
        tpl_data_y_op["groups"] = ""
    tpl_data_y_op["y1"] = {}
    tpl_data_y_op["y1"]["name"]  = y1_label
    tpl_data_y_op["y1"]["range"] = y1_range
    tpl_data_y_op["y2"] = {}
    tpl_data_y_op["y2"]["name"]  = y2_label if y2_label is not None else ''
    tpl_data_y_op["y2"]["range"] = y2_range if y2_range is not None else ''
    tpl_opt = {}
    tpl_opt["padding"] = [{"name":x, "val":y} for x, y in padding.items()] if padding is not None else []
    tpl = """
    // グラフ描画
    var chart = c3.generate({
    padding: {
        {%- for dictwk in tpl_opt.padding %}
        {{dictwk.name}}: {{dictwk.val}},
        {%- endfor %}
    },
    data: {
        xs: {
            {%- for dictwk in tpl_data_y %}
            '{{dictwk.name}}':'data_x',
            {%- endfor %}
        },
        columns: [
            ['data_x', {% for x in tpl_data_x.data %}'{{x}}',{% endfor %}],
            {%- for dictwk in tpl_data_y %}
            ['{{dictwk.name}}', {% for x in dictwk.data %}{{x}},{% endfor %}],
            {%- endfor %}
        ],
        {%- if tpl_data_y_op.type != '' %}
        type: '{{tpl_data_y_op.type}}',
        {%- endif %}
        types: {
            {%- for dictwk in tpl_data_y %}
            {%- if dictwk.type != '' %}
            {{dictwk.name}}:'{{dictwk.type}}',
            {%- endif  %}
            {%- endfor %}
        },
        {%- if tpl_data_y_op.groups != '' %}
        groups: [
            [{{tpl_data_y_op.groups}}],
        ],
        {%- endif %}
        {%- if tpl_data_y_op.y2.name != '' %}
        axes: {
            {%- for dictwk in tpl_data_y %}
            {{dictwk.name}}:'{{dictwk.y1ory2}}',
            {%- endfor %}
        },
        {%- endif %}
        colors: {
            {%- for dictwk in tpl_data_y %}
            {%- if dictwk.color != '' %}
            {{dictwk.name}}:'{{dictwk.color}}',
            {%- endif  %}
            {%- endfor %}
        },
        order: null,
        hide: [],
        labels: true,
    },
    axis: {
        x: {
            type: '{{tpl_data_x.type}}',
            {%- if tpl_data_x.type == 'category' %}
            category: [{% for x in tpl_data_x.data %}'{{x}}',{% endfor %}],
            {%- endif %}
            {%- if tpl_data_x.type == 'timeseries' %}
            tick: {
                format: '%Y-%m-%d',
                culling: {
                    max: 5, // the number of tick texts will be adjusted to less than this value
                },
            },
            {%- endif %}
            label: {
                text: '{{tpl_data_x.name}}',
                position: 'outer-center',
            },
        },
        y: {
            min:{{tpl_data_y_op.y1.range[0]}},
            max:{{tpl_data_y_op.y1.range[1]}},
            padding: {top: 0, bottom: 0},
            label: {
                text: '{{tpl_data_y_op.y1.name}}',
                position: 'outer-middle',
            },
        },
        {%- if tpl_data_y_op.y2.name != '' %}
        y2: {
            min:{{tpl_data_y_op.y2.range[0]}},
            max:{{tpl_data_y_op.y2.range[1]}},
            padding: {top: 0, bottom: 0},
            show: true,
            label: {
                text: '{{tpl_data_y_op.y2.name}}',
                position: 'outer-middle',
            },
        },
        {%- endif %}
    },
    grid: {
        y: {
            lines: [
                {%- for dictwk in lines %}
                {value: {{dictwk.data}}, text: '{{dictwk.text}}', axis:'{{dictwk.axis}}'},
                {%- endfor %}
            ],
        },
    },
    legend: {
    },
    size: {
    },
    bindto: '{{bindto}}'
    });
    """
    template = Template(tpl)
    html = template.render(tpl_data_x=tpl_data_x, tpl_data_y=tpl_data_y, tpl_data_y_op=tpl_data_y_op, lines=lines, bindto=bindto, tpl_opt=tpl_opt)
    return html


def create_datatables(df: pd.DataFrame, table_id: str, columns: List[str], data_name: dict, data_type: dict={}, data_url: dict={}, add_class: dict={}) -> str:
    """
    dataframe を input として datatables を書く
    Params::
        df: input dataframe
        table_id: table tag の html id
    """

    # template に埋め込む変数を定義していく
    datahead = []
    for x in columns:
        datahead.append(data_name[x])
    datatable = []
    for index in df.index:
        dictwk = {}
        dictwk["data"] = []
        for x in columns:
            dictwk["data"].append({
                "val": df.loc[index, x], "class":add_class[x] if add_class.get(x) is not None else "", 
                "url":df.loc[index, data_url[x]] if data_type.get(x) is not None and data_type.get(x) == "url" else ''
            })
        datatable.append(dictwk)

    tpl = """
    <table id="{{table_id}}" class="table table-bordered">
        <thead>
            <tr>
                {%- for x in datahead %}
                <th>{{ x }}</th>
                {%- endfor %}
            </tr>
        </thead>
        <tbody>
            {%- for dictwk in datatable %}
            <tr>
                {%- for dictwkwk in dictwk.data %}
                {%- if dictwkwk.url != '' %}
                <td><a href="{{ dictwkwk.url }}"><span class="d-inline-block text-truncate {{dictwkwk.class}}">{{ dictwkwk.val }}</span></a></td>
                {%- else %}
                <td><span class="d-inline-block text-truncate {{dictwkwk.class}}">{{ dictwkwk.val }}</span></td>
                {%- endif %}
                {%- endfor %}
            </tr>
            {%- endfor %}
        </tbody>
        <tfoot>
        </tfoot>
    </table>
    """
    template = Template(tpl)
    html = template.render(table_id=table_id, datahead=datahead, datatable=datatable)
    return html


def create_bs4table(
    df: pd.DataFrame, rows: List[str], columns: List[str], 
    row_names: dict=None, col_names: dict=None,
    table_id: str='', colors: List[str]=None, color_axis: int=-1
) -> str:
    df = df.loc[rows, columns].copy()
    colhead = df.columns.tolist() if col_names is None else [col_names[x] for x in df.columns]
    rowhead = df.index.  tolist() if row_names is None else [row_names[x] for x in df.index  ]
    values  = df.values. tolist()

    tpl = """
    <table {% if table_id != '' %}id="{{table_id}}"{% endif %} class="table table-bordered table-responsive-md">
        <thead>
            <tr>
                <th></th>
                {%- for x in colhead %}
                <th {% if color_axis == 1 %}{% if colors[loop.index0] != '' %}class="table-{{colors[loop.index0]}}"{% endif %}{% endif %}>{{ x }}</th>
                {%- endfor %}
            </tr>
        </thead>
        <tbody>
            {%- for listwk in values %}
            <tr {% if color_axis == 0 %}{% if colors[loop.index0] != '' %}class="table-{{colors[loop.index0]}}"{% endif %}{% endif %}>
                <th>{{rowhead[loop.index0]}}</th>
                {%- for x in listwk %}
                <td {% if color_axis == 1 %}{% if colors[loop.index0] != '' %}class="table-{{colors[loop.index0]}}"{% endif %}{% endif %}>{{x}}</td>
                {%- endfor %}
            </tr>
            {%- endfor %}
        </tbody>
        <tfoot>
        </tfoot>
    </table>
    """
    template = Template(tpl)
    html = template.render(table_id=table_id, colhead=colhead, rowhead=rowhead, values=values, colors=colors, color_axis=color_axis)
    return html
    
