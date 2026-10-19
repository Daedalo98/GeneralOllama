from __future__ import annotations

import streamlit as st
from streamlit_calendar import calendar

import json
import os
import tempfile
import uuid
import copy

from pathlib import Path
from datetime import date, datetime, time, timedelta, timezone
from typing import Any


# ==========================================
# TO DO CHECKLIST MANAGEMENT
# ==========================================

# The JSON file is stored relative to the directory where Streamlit is run.
TODO_FILE = Path("./todo.json")

def save_todos():
    """Write the current todo list to disk using an atomic file replacement."""
    # Create the parent directory if the file is later moved into a subfolder.
    TODO_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file in the same directory so os.replace is atomic.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=TODO_FILE.parent,
            delete=False,
        ) as temp_file:
            json.dump(
                st.session_state.todos,
                temp_file,
                indent=2,
                ensure_ascii=False,
            )
            temp_file.write("\n")
            temp_path = temp_file.name

        # Replacing the destination avoids leaving a partly written JSON file.
        os.replace(temp_path, TODO_FILE)
    finally:
        # Clean up the temporary file if an error occurred before replacement.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def load_todos():
    """Load todos from disk, or start with an empty list if no file exists."""
    if not TODO_FILE.exists():
        return []

    try:
        with TODO_FILE.open("r", encoding="utf-8") as todo_file:
            data = json.load(todo_file)

        # Ensure the file contains the expected top-level structure.
        if not isinstance(data, list):
            raise ValueError("The JSON file should contain a list of todos.")

        # Keep only dictionary entries with the fields this app requires.
        required_fields = {"id", "text", "checked", "note", "created_at"}
        valid_todos = [
            todo
            for todo in data
            if isinstance(todo, dict) and required_fields.issubset(todo)
        ]

        if len(valid_todos) != len(data):
            st.warning("Some entries in todo.json were invalid and were skipped.")

        return valid_todos

    except (json.JSONDecodeError, OSError, ValueError) as error:
        # Do not overwrite a file that could not be read without telling the user.
        st.error(f"Could not load {TODO_FILE}: {error}")
        st.stop()

def find_todo(todo_id):
    """Return the todo with this ID, or None if it no longer exists."""
    return next(
        (todo for todo in st.session_state.todos if todo["id"] == todo_id),
        None,
    )

def add_todo():
    """Create a todo from the add form and save it immediately."""
    text = st.session_state.new_todo_text.strip()
    if not text:
        return

    st.session_state.todos.append(
        {
            "id": str(uuid.uuid4()),
            "text": text,
            "checked": False,
            "note": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_todos()

    # Clear the form input before Streamlit redraws the page.
    st.session_state.new_todo_text = ""

def update_todo_text(todo_id):
    """Save an edited task title."""
    todo = find_todo(todo_id)
    if todo:
        todo["text"] = st.session_state[f"text_{todo_id}"].strip()
        save_todos()

def update_todo_checked(todo_id):
    """Save a checkbox change."""
    todo = find_todo(todo_id)
    if todo:
        todo["checked"] = st.session_state[f"checked_{todo_id}"]
        save_todos()

def update_todo_note(todo_id):
    """Save an edited note; store an empty note as None."""
    todo = find_todo(todo_id)
    if todo:
        note = st.session_state[f"note_{todo_id}"].strip()
        todo["note"] = note or None
        save_todos()

def delete_todo(todo_id):
    """Delete one todo and save the updated list."""
    st.session_state.todos = [
        todo for todo in st.session_state.todos if todo["id"] != todo_id
    ]
    save_todos()

def delete_checked_todos():
    """Delete all checked todos after the user confirms."""
    st.session_state.todos = [
        todo for todo in st.session_state.todos if not todo["checked"]
    ]
    save_todos()

    # Reset the confirmation checkbox after completing the deletion.
    st.session_state.confirm_delete_checked = False



# ==========================================
# TO-DO LIST MANAGEMENT
# ==========================================

# Load data once per Streamlit session. Streamlit reruns the script after
# interactions, so session state prevents reloading over the current UI state.
if "todos" not in st.session_state:
    st.session_state.todos = load_todos()

st.markdown("# To-do list")
with st.expander("Hide od Show To-Dos", expanded=True, width="stretch", icon="🗃️"):
    # The form makes Enter and the button submit the new task together.
    with st.form("add_todo_form", clear_on_submit=False):
        st.text_input("New task", key="new_todo_text")
        st.form_submit_button("Add todo", on_click=add_todo)


    # Filtering and ordering affect only the display, not the saved todo data.
    filter_choice = st.selectbox(
        "Show",
        ["All", "Active", "Checked"],
    )

    sort_choice = st.selectbox(
        "Order by",
        ["Newest first", "Oldest first", "Task name"],
    )

    visible_todos = st.session_state.todos.copy()

    if filter_choice == "Active":
        visible_todos = [todo for todo in visible_todos if not todo["checked"]]
    elif filter_choice == "Checked":
        visible_todos = [todo for todo in visible_todos if todo["checked"]]

    if sort_choice == "Newest first":
        visible_todos.sort(key=lambda todo: todo["created_at"], reverse=True)
    elif sort_choice == "Oldest first":
        visible_todos.sort(key=lambda todo: todo["created_at"])
    else:
        visible_todos.sort(key=lambda todo: todo["text"].casefold())


    # Bulk deletion is available only when there are checked todos.
    checked_count = sum(todo["checked"] for todo in st.session_state.todos)

    if checked_count:
        st.checkbox(
            f"I confirm deleting all {checked_count} checked todo(s)",
            key="confirm_delete_checked",
        )
        st.button(
            "Delete all checked",
            disabled=not st.session_state.get("confirm_delete_checked", False),
            on_click=delete_checked_todos,
        )


    if not visible_todos:
        st.info("No todos match this filter.")
    else:
        for todo in visible_todos:
            todo_id = todo["id"]

            # The title input is the editable task text. Checked tasks are also
            # shown with a check mark and strikethrough styling.
            title = todo["text"]
            if todo["checked"]:
                title = f"✅ ~~{title}~~"

            with st.expander(title, expanded=False):
                left_column, right_column = st.columns([5, 1])

                with left_column:
                    st.checkbox(
                        "Checked",
                        value=todo["checked"],
                        key=f"checked_{todo_id}",
                        on_change=update_todo_checked,
                        args=(todo_id,),
                    )

                    st.text_input(
                        "Task",
                        value=todo["text"],
                        key=f"text_{todo_id}",
                        on_change=update_todo_text,
                        args=(todo_id,),
                    )

                    if todo["note"] is not None:
                        st.text_area(
                            "Note",
                            value=todo["note"],
                            key=f"note_{todo_id}",
                            on_change=update_todo_note,
                            args=(todo_id,),
                        )
                    else:
                        # Keep the note editor hidden until the user asks for it.
                        add_note_key = f"show_note_{todo_id}"
                        if add_note_key not in st.session_state:
                            st.session_state[add_note_key] = False

                        if not st.session_state[add_note_key]:
                            if st.button("Add note", key=f"add_note_{todo_id}"):
                                st.session_state[add_note_key] = True
                                st.rerun()
                        else:
                            st.text_area(
                                "Note",
                                value="",
                                key=f"note_{todo_id}",
                                on_change=update_todo_note,
                                args=(todo_id,),
                                placeholder="Write a note...",
                            )

                with right_column:
                    st.button(
                        "Delete",
                        key=f"delete_{todo_id}",
                        on_click=delete_todo,
                        args=(todo_id,),
                    )


# ==========================================
# CALENDAR MANAGEMENT
# ==========================================

# FullCalendar event dictionaries support fields such as id, title, start, end,
# allDay, color, backgroundColor, borderColor, textColor, groupId, and resourceId.
# Resource IDs are referenced by events through "resourceId".
RESOURCES: list[dict[str, str]] = [
    {"id": "room-a", "title": "Room A", "building": "Building 1"},
    {"id": "room-b", "title": "Room B", "building": "Building 1"},
    {"id": "room-c", "title": "Room C", "building": "Building 2"},
]

SAMPLE_EVENTS: list[dict[str, Any]] = [
    {
        "id": "planning",
        "title": "Planning session",
        "start": "2026-10-19T09:00:00",
        "end": "2026-10-19T10:00:00",
        "color": "#3D9DF3",
        "resourceId": "room-a",
        "extendedProps": {"description": "Weekly team planning."},
    },
    {
        "id": "deadline",
        "title": "Project deadline",
        "start": "2026-10-22",
        "allDay": True,
        "color": "#FF6C6C",
        "extendedProps": {"description": "Submit the project deliverables."},
    },
    {
        "id": "review",
        "title": "Design review",
        "start": "2026-10-20T13:00:00",
        "end": "2026-10-20T14:30:00",
        "color": "#3DD56D",
        "resourceId": "room-b",
        "extendedProps": {"description": "Review the latest designs."},
    },
]

COLORS = {
    "Blue": "#3D9DF3",
    "Green": "#3DD56D",
    "Orange": "#FFBD45",
    "Red": "#FF6C6C",
    "Purple": "#A970FF",
}


# Store the JSON file in the same folder as this Python script,
# regardless of the process's current working directory.
DATA_FILE = Path("/home/daedalo/projects/calendar_events.json")

def load_events() -> list[dict[str, Any]]:
    """Load events from disk, or return the sample events on first run."""
    if not DATA_FILE.exists():
        return copy.deepcopy(SAMPLE_EVENTS)

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Calendar JSON must contain a list of events.")

        return data

    except (json.JSONDecodeError, OSError, ValueError) as error:
        st.error(f"Could not load {DATA_FILE.name}: {error}")
        return copy.deepcopy(SAMPLE_EVENTS)

def save_events(events: list[dict[str, Any]]) -> None:
    """Write the current events to the JSON file."""
    temporary_file = DATA_FILE.with_suffix(".json.tmp")

    try:
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(events, file, indent=2, ensure_ascii=False)
            file.write("\n")

        # Replace the old file only after the new JSON is fully written.
        os.replace(temporary_file, DATA_FILE)

    except OSError as error:
        st.error(f"Could not save calendar data to {DATA_FILE}: {error}")

# -----------------------------------------------------------------------------
# Session-state helpers
# -----------------------------------------------------------------------------

def initialize_state() -> None:
    """Load saved events once for this Streamlit session."""
    if "events" not in st.session_state:
        st.session_state.events = load_events()

    if "selected_event" not in st.session_state:
        st.session_state.selected_event = None

    if "calendar_callback" not in st.session_state:
        st.session_state.calendar_callback = None

    if "calendar_revision" not in st.session_state:
        st.session_state.calendar_revision = 0


def make_event_id() -> str:
    """Return a unique string ID for a newly created event."""
    return str(uuid.uuid4())


def event_by_id(event_id: str | None) -> dict[str, Any] | None:
    """Find an event in the app's source-of-truth list."""
    if not event_id:
        return None

    return next(
        (event for event in st.session_state.events if str(event.get("id")) == event_id),
        None,
    )


def parse_event_datetime(value: str | None) -> datetime | None:
    """Convert an ISO date/datetime string into a Python datetime."""
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def as_iso(value: date | datetime, all_day: bool) -> str:
    """Format a widget value in the form FullCalendar expects."""
    if all_day:
        if isinstance(value, datetime):
            value = value.date()
        return value.isoformat()

    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time.min)

    return value.isoformat(timespec="minutes")


def normalized_event(event: dict[str, Any]) -> dict[str, Any]:
    """
    Convert FullCalendar's callback event object to a serializable dict.

    FullCalendar may return fields such as startStr/endStr and extendedProps.
    Keep the app's editable fields and preserve existing custom metadata.
    """
    old = event_by_id(str(event.get("id", ""))) or {}

    normalized = copy.deepcopy(old)
    normalized["id"] = str(event.get("id") or old.get("id") or make_event_id())
    normalized["title"] = event.get("title", old.get("title", "Untitled event"))
    normalized["start"] = event.get("startStr") or event.get("start") or old.get("start")
    normalized["end"] = event.get("endStr") or event.get("end") or old.get("end")
    normalized["allDay"] = bool(event.get("allDay", old.get("allDay", False)))

    # Preserve common styling/resource fields if provided by the callback.
    for field in (
        "color",
        "backgroundColor",
        "borderColor",
        "textColor",
        "groupId",
        "resourceId",
    ):
        if event.get(field) is not None:
            normalized[field] = event[field]

    # FullCalendar can return custom event data inside extendedProps.
    if isinstance(event.get("extendedProps"), dict):
        normalized["extendedProps"] = event["extendedProps"]

    return normalized


# In upsert_event(), after adding or replacing the event
def upsert_event(updated: dict[str, Any]) -> None:
    """Add or replace an event, then persist the updated list."""
    event_id = str(updated["id"])

    for index, existing in enumerate(st.session_state.events):
        if str(existing.get("id")) == event_id:
            st.session_state.events[index] = updated
            save_events(st.session_state.events)
            return

    st.session_state.events.append(updated)
    save_events(st.session_state.events)



def view_options(view_mode: str) -> dict[str, Any]:
    """Build FullCalendar options for the selected view."""
    common: dict[str, Any] = {
        "editable": True,            # Allow drag-and-drop and resizing.
        "selectable": True,           # Allow selecting a date/time range.
        "navLinks": True,             # Click day/week headings to navigate.
        "nowIndicator": True,         # Show the current time in time-based views.
        "dayMaxEvents": True,         # Collapse excess month-view events.
        "weekends": True,
        "selectMirror": True,         # Preview a selection while dragging.
        "height": "auto",
        "timeZone": "local",
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek,timeGridDay,listWeek",
        },
        "buttonText": {
            "today": "Today",
            "month": "Month",
            "week": "Week",
            "day": "Day",
            "list": "Agenda",
        },
    }

    if view_mode == "Month":
        common.update(
            {
                "initialView": "dayGridMonth",
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "dayGridDay,dayGridWeek,dayGridMonth",
                },
            }
        )

    elif view_mode == "Time grid":
        common.update(
            {
                "initialView": "timeGridWeek",
                "slotMinTime": "06:00:00",
                "slotMaxTime": "22:00:00",
                "slotDuration": "00:30:00",
                "allDaySlot": True,
            }
        )

    elif view_mode == "Agenda list":
        common.update(
            {
                "initialView": "listWeek",
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "listDay,listWeek,listMonth",
                },
            }
        )

    elif view_mode == "Timeline":
        common.update(
            {
                "initialView": "timelineWeek",
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "timelineDay,timelineWeek,timelineMonth",
                },
                "slotMinTime": "06:00:00",
                "slotMaxTime": "22:00:00",
            }
        )

    elif view_mode == "Resource timeline":
        common.update(
            {
                "initialView": "resourceTimelineDay",
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": (
                        "resourceTimelineDay,"
                        "resourceTimelineWeek,"
                        "resourceTimelineMonth"
                    ),
                },
                "resources": RESOURCES,
                "resourceGroupField": "building",
                "resourceAreaHeaderContent": "Rooms",
                "slotMinTime": "06:00:00",
                "slotMaxTime": "22:00:00",
            }
        )

    elif view_mode == "Multi-month":
        common.update(
            {
                "initialView": "multiMonthYear",
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "multiMonthYear,dayGridMonth",
                },
            }
        )

    return common


initialize_state()


# -----------------------------------------------------------------------------
# Sidebar: choose a calendar mode and filter
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Calendar settings")

    selected_view = st.selectbox(
        "View",
        [
            "Month",
            "Time grid",
            "Agenda list",
            "Timeline",
            "Resource timeline",
            "Multi-month",
        ],
        help=(
            "Resource timeline uses FullCalendar's resource views. "
            "Check that your installed component build supports the selected view."
        ),
    )

    show_weekends = st.checkbox("Show weekends", value=True)
    allow_editing = st.checkbox("Allow drag and resize", value=True)
    allow_selection = st.checkbox("Allow date/time selection", value=True)

    selected_color = st.selectbox("New event color", list(COLORS))
    color_filter = st.selectbox(
        "Filter events",
        ["All colors", *COLORS.keys()],
        help="Filters the events passed to the calendar.",
    )

    # Reset sample events
    if st.button("Reset sample events", use_container_width=True):
        st.session_state.events = copy.deepcopy(SAMPLE_EVENTS)
        save_events(st.session_state.events)
        st.session_state.selected_event = None
        st.session_state.calendar_revision += 1



# Build options after the sidebar controls are read.
options = view_options(selected_view)
options["weekends"] = show_weekends
options["editable"] = allow_editing
options["selectable"] = allow_selection

# A small example of additional FullCalendar options you may choose to enable:
# options["businessHours"] = {
#     "daysOfWeek": [1, 2, 3, 4, 5],
#     "startTime": "09:00",
#     "endTime": "17:00",
# }
#
# Other useful options to explore in FullCalendar's documentation include:
# initialDate, locale, firstDay, height, contentHeight, aspectRatio,
# eventDisplay, eventOrder, eventOverlap, validRange, hiddenDays, and
# resourceOrder. The component forwards calendar configuration via "options".


# -----------------------------------------------------------------------------
# Event creation UI: form submission or selecting a calendar range
# -----------------------------------------------------------------------------

st.subheader("Add an event")

with st.form("add_event_form", clear_on_submit=True):
    form_col1, form_col2, form_col3 = st.columns([2, 1, 1])

    with form_col1:
        new_title = st.text_input("Title", placeholder="e.g. Team meeting")
        new_description = st.text_input("Description", placeholder="Optional notes")

    with form_col2:
        new_start_date = st.date_input("Start date", value=date.today())
        new_all_day = st.checkbox("All-day event", value=False)

    with form_col3:
        new_start_time = st.time_input("Start time", value=time(9, 0))
        new_end_time = st.time_input("End time", value=time(10, 0))

    resource_names = {item["id"]: item["title"] for item in RESOURCES}
    resource_id = None
    if selected_view == "Resource timeline":
        resource_id = st.selectbox(
            "Room/resource",
            options=list(resource_names),
            format_func=lambda key: resource_names[key],
        )

    submitted = st.form_submit_button("Create event", type="primary")

if submitted:
    if not new_title.strip():
        st.error("Please enter an event title.")
    else:
        if new_all_day:
            start_value = new_start_date
            end_value = new_start_date + timedelta(days=1)
        else:
            start_value = datetime.combine(new_start_date, new_start_time)
            end_value = datetime.combine(new_start_date, new_end_time)

            # If the end time is earlier than the start time, treat it as next day.
            if end_value <= start_value:
                end_value += timedelta(days=1)

        new_event: dict[str, Any] = {
            "id": make_event_id(),
            "title": new_title.strip(),
            "start": as_iso(start_value, new_all_day),
            "end": as_iso(end_value, new_all_day),
            "allDay": new_all_day,
            "color": COLORS[selected_color],
            "extendedProps": {"description": new_description.strip()},
        }

        if resource_id:
            new_event["resourceId"] = resource_id

        # After creating an event
        st.session_state.events.append(new_event)
        save_events(st.session_state.events)
        st.success(f"Created “{new_title.strip()}”.")



# -----------------------------------------------------------------------------
# Calendar display
# -----------------------------------------------------------------------------

def matches_filter(event: dict[str, Any]) -> bool:
    """Apply the color filter to the calendar's displayed event list."""
    if color_filter == "All colors":
        return True

    return event.get("color") == COLORS[color_filter]


visible_events = [
    copy.deepcopy(event)
    for event in st.session_state.events
    if matches_filter(event)
]

custom_css = """
/* Make past events slightly transparent. */
.fc-event-past {
    opacity: 0.75;
}

/* Emphasize event titles. */
.fc-event-title {
    font-weight: 600;
}

/* Adjust the toolbar title. */
.fc-toolbar-title {
    font-size: 1.5rem;
}

/* Use a clear visual cue for today's date. */
.fc .fc-day-today {
    background-color: rgba(61, 157, 243, 0.10) !important;
}
"""

# callbacks opts the component into returning those callback payloads.
# "eventsSet" reports the calendar's current event set; "eventChange" reports
# changes such as dragging or resizing. "select", "dateClick", and "eventClick"
# support the interactions handled below.
callback_names = [
    "dateClick",
    "select",
    "eventClick",
    "eventChange",
    "eventsSet",
    "eventMouseEnter",
    "eventMouseLeave",
]

calendar_state = calendar(
    events=visible_events,
    options=options,
    custom_css=custom_css,
    callbacks=callback_names,
    # Keep this key stable to avoid unnecessary component resets.
    # Bump the revision to force a fresh component instance after Reset.
    key=f"interactive_calendar_{selected_view}_{st.session_state.calendar_revision}",
)


# -----------------------------------------------------------------------------
# Process calendar callbacks
# -----------------------------------------------------------------------------

if isinstance(calendar_state, dict):
    callback_name = calendar_state.get("callback")
    st.session_state.calendar_callback = calendar_state

    # Clicking an event selects it for the editor below.
    if callback_name == "eventClick":
        clicked = calendar_state.get("eventClick", {}).get("event", {})
        st.session_state.selected_event = normalized_event(clicked)

    # Dragging/resizing commonly arrives as eventChange with the updated event.
    elif callback_name == "eventChange":
        change = calendar_state.get("eventChange", {})
        changed = change.get("event")

        if isinstance(changed, dict):
            upsert_event(normalized_event(changed))
            st.session_state.selected_event = normalized_event(changed)
            st.toast("Event time or date updated.")

    # Selecting a range makes it available as the default for the creation form
    # in a production app. Here we display it as an example callback.
    elif callback_name == "select":
        selection = calendar_state.get("select", {})
        selected_start = selection.get("start")
        selected_end = selection.get("end")
        if selected_start:
            st.info(
                f"Selected range: {selected_start}"
                + (f" → {selected_end}" if selected_end else "")
            )

    # A dateClick callback identifies the clicked date/time and optionally a
    # resource when the active view is a resource view.
    elif callback_name == "dateClick":
        click = calendar_state.get("dateClick", {})
        clicked_date = click.get("date")
        if clicked_date:
            st.caption(f"Clicked date/time: {clicked_date}")

    # Mouse callbacks are opt-in; useful for custom hover/status behavior.
    elif callback_name in ("eventMouseEnter", "eventMouseLeave"):
        # Inspect calendar_state for the associated event and view if needed.
        pass

    # eventsSet can be useful for observing the component's in-memory events.
    # Avoid blindly replacing source data from this callback: it can also fire
    # during initialization and can contain only the currently loaded events.
    elif callback_name == "eventsSet":
        pass


# -----------------------------------------------------------------------------
# Selected-event editor and delete action
# -----------------------------------------------------------------------------

selected = st.session_state.selected_event
if selected:
    st.subheader("Edit selected event")

    selected_id = str(selected.get("id", ""))
    stored = event_by_id(selected_id) or selected
    event_start = parse_event_datetime(stored.get("start"))
    event_end = parse_event_datetime(stored.get("end"))
    is_all_day = bool(stored.get("allDay", False))
    extended = stored.get("extendedProps", {})
    description = (
        extended.get("description", "")
        if isinstance(extended, dict)
        else ""
    )

    with st.form(f"edit_event_{selected_id}"):
        edit_title = st.text_input(
            "Title",
            value=str(stored.get("title", "")),
        )
        edit_description = st.text_area("Description", value=description)

        edit_all_day = st.checkbox("All-day event", value=is_all_day)

        fallback_date = date.today()
        start_date_value = event_start.date() if event_start else fallback_date
        end_date_value = event_end.date() if event_end else start_date_value

        edit_start_date = st.date_input("Start date", value=start_date_value)
        edit_end_date = st.date_input("End date", value=end_date_value)

        default_start_time = event_start.time() if event_start else time(9, 0)
        default_end_time = event_end.time() if event_end else time(10, 0)
        edit_start_time = st.time_input("Start time", value=default_start_time)
        edit_end_time = st.time_input("End time", value=default_end_time)

        edit_color_name = st.selectbox(
            "Color",
            list(COLORS),
            index=next(
                (
                    i
                    for i, color in enumerate(COLORS.values())
                    if color == stored.get("color")
                ),
                0,
            ),
        )

        save_edit = st.form_submit_button("Save changes", type="primary")

    delete_col, clear_col = st.columns(2)

    if save_edit:
        if not edit_title.strip():
            st.error("The event title cannot be empty.")
        else:
            if edit_all_day:
                start_value = edit_start_date
                # FullCalendar treats all-day end dates as exclusive.
                end_value = max(edit_end_date, edit_start_date) + timedelta(days=1)
            else:
                start_value = datetime.combine(edit_start_date, edit_start_time)
                end_value = datetime.combine(edit_end_date, edit_end_time)
                if end_value <= start_value:
                    st.error("The event end must be after its start.")
                    st.stop()

            updated_event = copy.deepcopy(stored)
            updated_event.update(
                {
                    "title": edit_title.strip(),
                    "start": as_iso(start_value, edit_all_day),
                    "end": as_iso(end_value, edit_all_day),
                    "allDay": edit_all_day,
                    "color": COLORS[edit_color_name],
                    "extendedProps": {"description": edit_description.strip()},
                }
            )

            upsert_event(updated_event)  # This now saves to JSON as well.
            st.session_state.selected_event = updated_event
            st.success("Event updated.")
            st.rerun()

    with delete_col:
        if st.button("Delete selected event", type="secondary"):
            st.session_state.events = [
                event
                for event in st.session_state.events
                if str(event.get("id")) != selected_id
            ]

            # After deleting an event
            st.session_state.events = [
                event
                for event in st.session_state.events
                if str(event.get("id")) != selected_id
            ]
            save_events(st.session_state.events)
            st.session_state.selected_event = None
            st.session_state.calendar_revision += 1
            st.rerun()


    with clear_col:
        if st.button("Clear selection"):
            st.session_state.selected_event = None
            st.rerun()


# -----------------------------------------------------------------------------
# Debug / inspection panels
# -----------------------------------------------------------------------------

with st.expander("Callback payload (debugging)"):
    st.write(
        "The returned dictionary varies by callback. Inspect it here while "
        "developing your app."
    )
    st.json(calendar_state if isinstance(calendar_state, dict) else {})

with st.expander("Events currently stored in this session"):
    st.json(st.session_state.events)


