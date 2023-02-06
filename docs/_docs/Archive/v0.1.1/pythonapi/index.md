---
title: "Python API"
description: Python API
---

# Python API

The full API documentation is listed here. Our documentation structure is referenced from Pytorch.

## Note

One of the purposes of PyDec is to allow users to do tensor tracking based on the same API.
Therefore, most of the APIs we designed are based on the rule that performing a certain operation on a composition is equivalent to performing the same operation on each of the composition's components.

In addition, we have designed some APIs specifically for composition to manipulate it or to query related information in the component dimension.
These APIs usually start with `c_`.


## Quick Jump
<div class="section-index">
    <hr class="panel-line">
    {% for post in site.docs %}
        {% assign path_segments = post.url | split: "/" %}
        {% if path_segments[2] == "pythonapi"%}
            <div class="entry">
            <h5><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h5>
            <p>{{ post.description }}</p>
            </div>
        {% endif %}
    {% endfor %}
</div>