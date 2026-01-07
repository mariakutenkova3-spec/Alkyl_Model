import tkinter as tk
from tkinter import messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Alkyl_Consts import molar_masses, substances
from Alkyl_Calc import calculate, optimal_T


# текстовый отчет
def show_report(parent, reactions, start_mass, diff_mass, y0, solution, Q_tuple):
    (r_11, r_12, r_21, r_22, r_31, r_32, r_41, r_42, r_51, r_52) = reactions
    # Q_tuple = (Q_in, Q_react, Q_out, Q_balance, T1)
    Q_in, Q_react, Q_out, Q_balance, T1 = Q_tuple

    end_mass = float(np.dot(molar_masses, solution.y[:, -1]))
    diff_abs = start_mass - end_mass
    diff_rel = (diff_abs / start_mass) * 100 if start_mass != 0 else 0.0

    # очищаем область отчета
    for w in parent.winfo_children():
        w.destroy()

    # температура продуктов и теплота реакций
    tk.Label(
        parent,
        text=f"Температура продуктов: {T1:.3f} K",
        font=("Arial", 10, "bold")
    ).grid(row=0, column=0, columnspan=4, pady=(5, 2), sticky="w")

    tk.Label(
        parent,
        text=f"Суммарный тепловой эффект реакций: {Q_react:.3f} кДж",
        font=("Arial", 10, "bold")
    ).grid(row=1, column=0, columnspan=4, pady=(0, 5), sticky="w")

    spacer_row = 2

    # константы скоростей
    tk.Label(parent, text="Константы скоростей",
             font=("Arial", 10, "bold")).grid(
        row=spacer_row, column=0, columnspan=4, pady=(5, 2)
    )

    headers = ["Реакция", "k_прямой", "k_обратной"]
    for j, h in enumerate(headers):
        tk.Label(parent, text=h, borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=spacer_row + 1, column=j, sticky="nsew")

    reactions_names = ["Б+Э <-> ЭБ",
                       "Б+П <-> ПБ",
                       "МБ+Э <-> МЭБ",
                       "Б+ЭБ <-> ДФЭ+В",
                       "ЭБ+Э <-> ДЭБ"]
    k_pairs = [(r_11.k, r_12.k),
               (r_21.k, r_22.k),
               (r_31.k, r_32.k),
               (r_41.k, r_42.k),
               (r_51.k, r_52.k)]
    for i, (name, (k_f, k_b)) in enumerate(zip(reactions_names, k_pairs),
                                           start=spacer_row + 2):
        tk.Label(parent, text=name, borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=i, column=0, sticky="nsew")
        tk.Label(parent, text=f"{k_f:.6f}", borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=i, column=1, sticky="nsew")
        tk.Label(parent, text=f"{k_b:.6f}", borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=i, column=2, sticky="nsew")

    row_after_k = (spacer_row + 2) + len(reactions_names) + 1

    # материальный и тепловой балансы
    tk.Label(parent, text="Материальный и тепловой баланс", font=("Arial", 10, "bold")).grid(
        row=row_after_k, column=0, columnspan=4, pady=(10, 2)
    )

    headers_mb = ["Показатель", "Материальный баланс", "Тепловой баланс"]
    for j, h in enumerate(headers_mb):
        tk.Label(parent, text=h, borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=row_after_k + 1, column=j, sticky="nsew")

    r0 = row_after_k + 2

    tk.Label(parent, text="Начальное состояние", borderwidth=1, relief="solid",
             padx=4, pady=2).grid(row=r0, column=0, sticky="nsew")
    tk.Label(parent, text=f"{start_mass:.3f} кг/с", borderwidth=1, relief="solid",
             padx=4, pady=2).grid(row=r0, column=1, sticky="nsew")
    tk.Label(parent, text=f"{(Q_in + Q_react):.3f} кДж", borderwidth=1, relief="solid",
             padx=4, pady=2).grid(row=r0, column=2, sticky="nsew")

    tk.Label(parent, text="Конечное состояние", borderwidth=1, relief="solid",
             padx=4, pady=2).grid(row=r0 + 1, column=0, sticky="nsew")
    tk.Label(parent, text=f"{end_mass:.3f} кг/с", borderwidth=1, relief="solid",
             padx=4, pady=2).grid(row=r0 + 1, column=1, sticky="nsew")
    tk.Label(parent, text=f"{Q_out:.3f} кДж", borderwidth=1, relief="solid",
             padx=4, pady=2).grid(row=r0 + 1, column=2, sticky="nsew")

    tk.Label(parent, text="Отклонение, %", borderwidth=1, relief="solid",
             padx=4, pady=2).grid(row=r0 + 2, column=0, sticky="nsew")
    tk.Label(parent, text=f"{np.abs(diff_rel):.3f}",
             borderwidth=1, relief="solid", padx=4, pady=2).grid(row=r0 + 2, column=1, sticky="nsew")
    tk.Label(parent, text=f"{np.abs(Q_balance):.3f}",
             borderwidth=1, relief="solid", padx=4, pady=2).grid(row=r0 + 2, column=2, sticky="nsew")

    row_conc = r0 + 4
    tk.Label(parent, text="Количество каждого вещества, кмоль/с", font=("Arial", 10, "bold")).grid(
        row=row_conc, column=0, columnspan=4, pady=(10, 2)
    )
    #исходные и конечные концентрации

    headers_c = ["Вещество", "C_0", "C_конечная"]
    for j, h in enumerate(headers_c):
        tk.Label(parent, text=h, borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=row_conc + 1, column=j, sticky="nsew")

    for i, name in enumerate(substances):
        tk.Label(parent, text=name, borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=row_conc + 2 + i, column=0, sticky="nsew")
        tk.Label(parent, text=f"{y0[i]:.6f}", borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=row_conc + 2 + i, column=1, sticky="nsew")
        tk.Label(parent, text=f"{solution.y[i, -1]:.6f}", borderwidth=1, relief="solid",
                 padx=4, pady=2).grid(row=row_conc + 2 + i, column=2, sticky="nsew")

    for c in range(4):
        parent.grid_columnconfigure(c, weight=1)


def plot_results_on_axes(ax_top_left, ax_top_right, ax_bottom_left, ax_bottom_right,
                         solution, Ts, C_EB_end, y0):
    # очищаем 4 графика
    ax_top_left.clear()
    ax_top_right.clear()
    ax_bottom_left.clear()
    ax_bottom_right.clear()

    t = solution.t  # вектор временных точек

    #  C_EB(T) и T_опт
    ax_top_left.plot(Ts, C_EB_end, '-o', label='C_EB(T)', color='g')
    idx_opt = np.argmax(C_EB_end)
    T_opt = Ts[idx_opt]
    C_opt = C_EB_end[idx_opt]

    ax_top_left.axvline(T_opt, color='r', linestyle='--', label='T_опт')
    ax_top_left.scatter([T_opt], [C_opt], color='r')
    ax_top_left.text(
        1.02, 0.7,
        f"T_opt: {T_opt:.1f} K",
        fontsize=10, color='black', fontweight='bold',
        transform=ax_top_left.transAxes
    )

    ax_top_left.set_xlabel('T, K')
    ax_top_left.set_ylabel('Конечная C_EB')
    ax_top_left.set_title('Зависимость конечной C_EB от температуры')
    ax_top_left.grid(True)
    ax_top_left.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                       borderaxespad=0.)

    #  конверсия бензола и этилена
    i_B = 0
    i_E = 1

    C_B0 = y0[i_B]
    C_E0 = y0[i_E]

    if C_B0 > 0:
        X_B = (C_B0 - solution.y[i_B, :]) / C_B0
    else:
        X_B = np.zeros_like(t)

    if C_E0 > 0:
        X_E = (C_E0 - solution.y[i_E, :]) / C_E0
    else:
        X_E = np.zeros_like(t)

    ax_top_right.plot(t, X_B, label='X_B', color='tab:blue')
    ax_top_right.plot(t, X_E, label='X_E', color='tab:orange')
    ax_top_right.set_xlabel('Время, с')
    ax_top_right.set_ylabel('Степень конверсии')
    ax_top_right.set_ylim(0, 1)
    ax_top_right.set_title('Зависимость конверсии бензола и этилена от времени')
    ax_top_right.grid(True)
    ax_top_right.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.)

    # прямая реакция
    line_B,  = ax_bottom_left.plot(t, solution.y[0], label='C_B')
    line_EB, = ax_bottom_left.plot(t, solution.y[2], label='C_EB')
    line_E,  = ax_bottom_left.plot(t, solution.y[1], label='C_E')

    ax_bottom_left.set_xlabel('Время')
    ax_bottom_left.set_ylabel('Концентрация')
    ax_bottom_left.set_title('Изменение концентраций веществ прямой реакции')
    ax_bottom_left.grid(True)
    ax_bottom_left.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                          borderaxespad=0.)

    # побочные реакции
    comp = [3, 4, 5, 6, 7, 8, 9]
    side_lines = []
    for i in comp:
        ln, = ax_bottom_right.plot(t, solution.y[i], label=substances[i])
        side_lines.append(ln)

    ax_bottom_right.set_xlabel('Время')
    ax_bottom_right.set_ylabel('Концентрация')
    ax_bottom_right.set_title('Изменение концентраций веществ побочных реакций')
    ax_bottom_right.grid(True)
    ax_bottom_right.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                           borderaxespad=0.)

    main_lines = [line_B, line_EB, line_E]
    return main_lines, side_lines



def start_modeling(initial_params):
    root = tk.Tk()
    root.title("Моделирование процесса алкилирования бензола этан-этиленовой фракцией по технологии EBMAX")
    root.state('zoomed')

    frame_left = tk.Frame(root)
    frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

    frame_controls = tk.Frame(frame_left, bg="darkolivegreen", relief="raised", bd=2)
    frame_controls.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

    tk.Label(
        frame_controls,
        text="Параметры моделирования",
        font=("Arial", 12, "bold"),
        fg="white",
        bg="darkolivegreen"
    ).grid(row=0, column=0, columnspan=2, pady=(5, 10))

    labels = [
        "Давление p, атм:", "Мольная доля B:", "Мольная доля E:",
        "Мольная доля P:", "Мольная доля MB:",
        "Время реакции t, с:", "Температура T, K:", "Мольный расход G_M, кмоль/с:"
    ]

    for i, text in enumerate(labels, start=1):
        tk.Label(
            frame_controls,
            text=text,
            font=("Arial", 9, "bold"),
            fg="white",
            bg="darkolivegreen"
        ).grid(row=i, column=0, sticky="w", padx=10, pady=3)

    p_var = tk.StringVar(value=str(initial_params["p"]))
    cb_var = tk.StringVar(value=str(initial_params["c_b_0"]))
    ce_var = tk.StringVar(value=str(initial_params["c_e_0"]))
    cp_var = tk.StringVar(value=str(initial_params["c_p_0"]))
    cmb_var = tk.StringVar(value=str(initial_params["c_mb_0"]))
    t_var = tk.StringVar(value=str(initial_params["t_max"]))
    T_var = tk.StringVar(value=str(initial_params["T"]))
    G_M_var = tk.StringVar(value=str(initial_params["G_M"]))

    entries = [p_var, cb_var, ce_var, cp_var, cmb_var, t_var, T_var, G_M_var]

    for i, var in enumerate(entries, start=1):
        entry = tk.Entry(
            frame_controls,
            textvariable=var,
            font=("Arial", 10),
            relief="solid",
            bd=1,
            bg="linen",
            insertbackground="black",
            selectbackground="darkolivegreen"
        )
        entry.grid(row=i, column=1, padx=10, pady=3, sticky="ew")

    frame_controls.grid_columnconfigure(1, weight=1)

    report_container = tk.Frame(frame_left, borderwidth=1, relief="sunken", bg="beige")
    report_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 0))

    report_canvas = tk.Canvas(
        report_container,
        bg="beige",
        highlightthickness=0,
        borderwidth=0
    )
    report_scrollbar = tk.Scrollbar(report_container, orient="vertical", command=report_canvas.yview)
    report_canvas.configure(yscrollcommand=report_scrollbar.set)

    report_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    report_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    report_frame = tk.Frame(report_canvas, bg="beige")
    report_canvas_window = report_canvas.create_window((0, 0), window=report_frame, anchor="nw")

    def _update_report_scrollregion(event):
        report_canvas.configure(scrollregion=report_canvas.bbox("all"))
        canvas_width = report_canvas.winfo_width()
        report_canvas.itemconfig(report_canvas_window, width=canvas_width)

    report_frame.bind("<Configure>", _update_report_scrollregion)

    def _on_mousewheel(event):
        report_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    report_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    frame_right = tk.Frame(root)
    frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    frame_plot = tk.Frame(frame_right)
    frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    fig = Figure(figsize=(8, 6), dpi=100)
    ax_top_left = fig.add_subplot(2, 2, 1)
    ax_top_right = fig.add_subplot(2, 2, 2)
    ax_bottom_left = fig.add_subplot(2, 2, 3)
    ax_bottom_right = fig.add_subplot(2, 2, 4)

    fig.subplots_adjust(
        left=0.07,
        right=0.9,
        bottom=0.12,
        top=0.92,
        wspace=0.55,
        hspace=0.40
    )

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    frame_checks = tk.Frame(frame_right, bg="beige")
    frame_checks.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 5))

    main_lines = []
    side_lines = []
    main_checks = {}
    side_checks = {}
    T_opt_label = None

    def update_visibility():
        ys_main = []
        for name, line in zip(['C_B', 'C_EB', 'C_E'], main_lines):
            var = main_checks.get(name)
            visible = var.get() if var is not None else True
            line.set_visible(visible)
            if visible:
                ys_main.append(line.get_ydata())
        if ys_main:
            y_all = np.concatenate(ys_main)
            ax_bottom_left.set_ylim(y_all.min() * 0.95, y_all.max() * 1.05)
        else:
            ax_bottom_left.set_ylim(0, 1)

        ys_side = []
        side_names = [substances[i] for i in [3, 4, 5, 6, 7, 8, 9]]
        for name, line in zip(side_names, side_lines):
            var = side_checks.get(name)
            visible = var.get() if var is not None else True
            line.set_visible(visible)
            if visible:
                ys_side.append(line.get_ydata())
        if ys_side:
            y_all2 = np.concatenate(ys_side)
            ax_bottom_right.set_ylim(y_all2.min() * 0.95, y_all2.max() * 1.05)
        else:
            ax_bottom_right.set_ylim(0, 1)

        canvas.draw_idle()

    def build_checkboxes():
        for w in frame_checks.winfo_children():
            w.destroy()

        tk.Label(frame_checks, text="Показать:", font=("Arial", 11, "bold"), bg="beige").pack(side=tk.LEFT, padx=(5, 10))

        for name in ['C_B', 'C_EB', 'C_E']:
            var = tk.BooleanVar(value=True)
            main_checks[name] = var
            cb = tk.Checkbutton(
                frame_checks, text=name, variable=var,
                command=update_visibility, font=("Arial", 11),
                bg="beige"
            )
            cb.pack(side=tk.LEFT, padx=5, pady=2)

        tk.Label(frame_checks, text="|", font=("Arial", 11), bg="beige").pack(side=tk.LEFT, padx=5)

        for i in [3, 4, 5, 6, 7, 8, 9]:
            name = substances[i]
            var = tk.BooleanVar(value=True)
            side_checks[name] = var
            cb = tk.Checkbutton(
                frame_checks, text=name, variable=var,
                command=update_visibility, font=("Arial", 11),
                bg="beige"
            )
            cb.pack(side=tk.LEFT, padx=5, pady=2)

    def result():
        nonlocal main_lines, side_lines

        try:
            p = float(p_var.get())
            c_b_0 = float(cb_var.get())
            c_e_0 = float(ce_var.get())
            c_p_0 = float(cp_var.get())
            c_mb_0 = float(cmb_var.get())
            t_max = float(t_var.get())
            T = float(T_var.get())
            G_M = float(G_M_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте числовые значения параметров.")
            return

        c_b_0 *= G_M
        c_e_0 *= G_M
        c_p_0 *= G_M
        c_mb_0 *= G_M

        # оптимальная температура
        T_opt_res, Ts, C_EB_end = optimal_T(p, c_b_0, c_e_0, c_p_0, c_mb_0, t_max)

        # расчет процесса
        solution, start_mass, diff_mass, reactions_tuple, y0, Q_tuple_full = calculate(
            p, c_b_0, c_e_0, c_p_0, c_mb_0, t_max, T, G_M
        )
        Q_in, Q_react, Q_out_final, Q_balance_final, T1, Q_diff_time = Q_tuple_full
        Q_tuple_report = (Q_in, Q_react, Q_out_final, Q_balance_final, T1)

        show_report(report_frame, reactions_tuple, start_mass, diff_mass, y0,
                    solution, Q_tuple_report)

        main_lines, side_lines = plot_results_on_axes(
            ax_top_left, ax_top_right,
            ax_bottom_left, ax_bottom_right,
            solution, Ts, C_EB_end, y0
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.9,
            bottom=0.12,
            top=0.92,
            wspace=0.55,
            hspace=0.40
        )
        canvas.draw()

        build_checkboxes()
        update_visibility()

    tk.Button(
        frame_controls,
        text="Расчет и построение",
        command=result,
        bg="beige",
        fg="black",
        activebackground="beige",
        relief="raised"
    ).grid(row=len(labels)+1, column=0, columnspan=2, pady=8)

    root.mainloop()
