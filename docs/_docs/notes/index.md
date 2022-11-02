---
title: "Notes"
description: Notes
---

# Notes

The developer notes provide a general description of each module as well as some development instructions.

## Quick Jump
<div class="section-index">
    <hr class="panel-line">
    {% for post in site.docs %}
        {% assign path_segments = post.url | split: "/" %}
        {% if path_segments[2] == "notes"%}
            <div class="entry">
            <h5><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h5>
            <p>{{ post.description }}</p>
            </div>
        {% endif %}
    {% endfor %}
</div>