---
title: Tutorials
description: Getting started with PyDec.
---

# Tutorials
TODO

## Quick Jump
<div class="section-index">
    <hr class="panel-line">
    {% for post in site.docs %}
        {% assign path_segments = post.url | split: "/" %}
        {% if path_segments[2] == "tutorials"%}
            <div class="entry">
            <h5><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h5>
            <p>{{ post.description }}</p>
            </div>
        {% endif %}
    {% endfor %}
</div>