#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

from nomad.metainfo import (Quantity, SubSection, Section, MSection)
from nomad.datamodel.data import ArchiveSection


class Reactant(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    amount = Quantity(type=np.dtype(np.float64), shape=['*'])


class Feed(ArchiveSection):
    m_def = Section(a_plot=[
        {
            "label": "Feed", 'x': 'runs', 'y': ['reactants/:/amount'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}}])

    space_velocity = Quantity(
        type=np.dtype(np.float64), shape=['*'], unit='1/hour')

    flow_rates = Quantity(
        type=np.dtype(np.float64), shape=['*'], unit='mL/min')

    runs = Quantity(type=np.dtype(np.float64), shape=['*'])

    time_on_stream = Quantity(
        type=np.dtype(np.float64), shape=['*']
    )

    reactants = SubSection(section_def=Reactant, repeats=True)


class Product(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    selectivity = Quantity(type=np.dtype(np.float64), shape=['*'])


class Conversion(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    conversion_product_based = Quantity(type=np.dtype(np.float64), shape=['*'])
    conversion_reactant_based = Quantity(type=np.dtype(np.float64), shape=['*'])


class Rate(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    reaction_rate = Quantity(type=np.dtype(np.float64), shape=['*'], unit='mmol/g/hour')


class CatalyticReactionData(ArchiveSection):
    m_def = Section(a_plot=[
        {
            "label": "Selectivity [%]",
            'x': 'runs',
            'y': ['products/:/selectivity'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Conversion X [%]",
            'x': 'runs',
            'y': ['conversion/:/conversion_product_based'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Conversion x_r ",
            'x': 'runs',
            'y': ['conversion/:/conversion_reactant_based'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Carbon Balance",
            'x': 'runs',
            'y': ['c_balance'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Temperature",
            'x': 'runs',
            'y': ['temperature'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "S_X plot",
            'x': ['conversion/0:1/conversion_product_based'],
            'y': ['products/:/selectivity'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True},
            "lines": [
                {"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"}]
        },
        {
            "label": "S_X plot 2",
            'x': ['conversion/1:2/conversion_product_based'],
            'y': ['products/:/selectivity'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True},
            "lines": [{"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"},
                      {"mode": "markers"}]
        },
        {
            "label": "rate",
            'x': 'runs',
            'y': ['rates/:/reaction_rate'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Temp vs. rate",
            'x': 'temperature',
            'y': ['rates/:/reaction_rate'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
    ]
    )

    temperature = Quantity(
        type=np.dtype(
            np.float64), shape=['*'], unit='°C')

    temperature_min = Quantity(
        type=np.dtype(
            np.float64), shape=[], unit='°C')

    temperature_max = Quantity(
        type=np.dtype(
            np.float64), shape=[], unit='°C')

    c_balance = Quantity(
        type=np.dtype(
            np.float64), shape=['*'])

    runs = Quantity(type=np.dtype(np.float64), shape=['*'])
    time = Quantity(type=np.dtype(np.float64), shape=['*'], unit='s')

    products = SubSection(section_def=Product, repeats=True)

    conversion = SubSection(section_def=Conversion, repeats=True)
    rates = SubSection(section_def=Rate, repeats=True)


class ECatalyticReaction(MSection):

    reaction = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    electrolyte = Quantity(type=str, a_eln=dict(component='StringEditQuantity',
                                                props=dict(suggestions=['Water', 'Base',
                                                                        'Acid'])))

    pH = Quantity(type=np.dtype(np.float64),
                  unit=("pH"),
                  a_eln=dict(
                      component='NumberEditQuantity', defaultDisplayUnit='pH'))

    potential = Quantity(
        type=np.dtype(np.float64),
        unit=("mV"),
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='mV'))

    electrode_support = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    cell_type = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    counter_electrode = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    feed = SubSection(section_def=Feed)
    reactants = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    def normalize(self, archive, logger):
        super(ECatalyticReaction, self).normalize(archive, logger)
