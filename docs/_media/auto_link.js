var link_render = function () {
    function parser(text) {
        let out = { key: null, short: 0, with_parentheses: true };
        let parsed_out = {};
        if (text.startsWith("{") && text.endsWith("}")) {
            parsed_out = JSON.parse(text);
        } else {
            values = text.split(" ");
            for (idx in values) {
                value = values[idx];
                if (value.startsWith("short")) {
                    if (value == "short") {
                        parsed_out["short"] = -1;
                    } else {
                        parsed_out["short"] = parseInt(value.split(":")[1]);
                    }
                } else if (value.startsWith("with_parentheses")) {
                    parsed_out["with_parentheses"] = (value.split(":")[1] == 'true');
                } else {
                    parsed_out["key"] = value;
                }
            }
        }
        out = Object.assign(out, parsed_out);
        return out;
    };
    function get_version() {
        const web_href = window.location.href;
        const web_origin = window.location.origin;
        const pathname = web_href.substring(web_origin.length + 3); // web_origin.length + "/#/"
        let version = "";
        if (pathname.startsWith("v")) {
            version = pathname.substring(0, pathname.indexOf("/"));
        }
        return version;
    }
    let mustache_data = null;
    let data_url = "_media/vars.json";
    Docsify.get(data_url, true).then((response) => {
        mustache_data = JSON.parse(response);
    }, (error) => {
        console.log(error);
    });
    return function (text, render) {
        parsed_data = parser(text);
        if (parsed_data.key == null || mustache_data == null) {
            return text;
        }
        const path = mustache_data[parsed_data.key]
        const version = get_version()
        let link_label = parsed_data.key;
        if (parsed_data.short != 0) {
            link_label = link_label.split(".").slice(parsed_data.short).join(".");
        }
        if (parsed_data.with_parentheses) {
            link_label = link_label + "()";
        }
        rendered_link = `[\`${link_label}\`](${version}/${path})`;
        return rendered_link;
    };
}