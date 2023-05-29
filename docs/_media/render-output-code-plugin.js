// this script transfer comment code block to output style
(function () {
    render_plugin = function (hook, vm) {
        hook.doneEach(function () {
            var code_comments = $(".token.triple-quoted-string.string")
            for (var i = 0; i < code_comments.length; i++) {
                if (code_comments[i].innerHTML.startsWith("\"\"\"\n") || code_comments[i].innerHTML.startsWith("\'\'\'\n")) {
                    var removed_str = code_comments[i].innerHTML.slice(4, -4)
                    code_comments[i].innerHTML = removed_str
                    code_comments[i].className = 'token comment'
                }
            }
        });
    };

    // Add plugin to docsify's plugin array
    $docsify = $docsify || {};
    $docsify.plugins = [].concat($docsify.plugins || [], render_plugin);
})();