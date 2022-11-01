---
layout: page
title: Documentation
permalink: /docs/
---

# Documentation

Welcome to the {{ site.title }} Documentation pages! Here you can quickly jump to a 
particular page.

<div class="section-index">
    <hr class="panel-line">
    {% for post in site.docs  %}
    {% assign path_segment = post.url | split: "/" | slice: 1 %}
    {% if path_segment != "Archive" %}
    <div class="entry">
    <h5><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h5>
    <p>{{ post.description }}</p>
    <p>{{ post.url }}</p>
    <p>{{ path_segment }}</p>
    </div>
    {% endif %}
    {% endfor %}
</div>